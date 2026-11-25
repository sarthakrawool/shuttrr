// metal4_bench: does the M5 GPU's Metal 4 TensorOps path (mpp::tensor_ops::matmul2d, i.e. the
// "neural accelerators") beat a plain simdgroup_matrix GEMM, and how close does it get to what
// MLX already reaches? Standalone; needs only macOS + the Metal framework.
//
//   swiftc -O main.swift -o metal4_bench        (build_and_run.sh does this)
//   ./metal4_bench --yes --out result.json [--seconds 5] [--max-gb 8] [--quick]
//
// Refuses to run without --yes. Allocations are capped (default and hard maximum 8 GB).
// Changes no system setting. Writes JSON to stdout and to --out. Progress goes to stderr.
// Written on Linux without a Swift or Metal compiler: see README.md for what is unverified.

import Foundation
import Metal

// MARK: - Options

struct Options {
    var yes = false
    var out: String? = nil
    var seconds = 5.0
    var maxGB = 8.0
    var shaderPath: String? = nil
    var dtypes: [String] = ["fp16", "bf16"]
    var int8 = true
    var quick = false
    var shapes: [String]? = nil
    var qkTile = 4160
    var pvTile = 12480
}

let usage = """
usage: metal4_bench --yes [--out FILE] [--seconds S] [--max-gb G] [--shader kernels.metal]
                    [--dtypes fp16,bf16] [--no-int8] [--quick] [--shapes square,qk,pv]
                    [--qk-tile ROWS] [--pv-tile ROWS]
  --yes        required: confirms you want to use the GPU for a few minutes
  --out        also write the JSON result to FILE
  --seconds    timed budget per case (default 5)
  --max-gb     allocation cap in GB (default 8, cannot be raised above 8)
  --quick      1 s per case, square shape only (smoke test)
"""

func parseOptions() -> Options {
    var o = Options()
    var args = Array(CommandLine.arguments.dropFirst())
    func next(_ name: String) -> String {
        guard !args.isEmpty else { FileHandle.standardError.write("missing value for \(name)\n".data(using: .utf8)!); exit(2) }
        return args.removeFirst()
    }
    while !args.isEmpty {
        let a = args.removeFirst()
        switch a {
        case "--yes": o.yes = true
        case "--out": o.out = next(a)
        case "--seconds": o.seconds = Double(next(a)) ?? 5.0
        case "--max-gb": o.maxGB = Double(next(a)) ?? 8.0
        case "--shader": o.shaderPath = next(a)
        case "--dtypes": o.dtypes = next(a).split(separator: ",").map(String.init)
        case "--no-int8": o.int8 = false
        case "--quick": o.quick = true
        case "--shapes": o.shapes = next(a).split(separator: ",").map(String.init)
        case "--qk-tile": o.qkTile = Int(next(a)) ?? 4160
        case "--pv-tile": o.pvTile = Int(next(a)) ?? 12480
        case "-h", "--help": print(usage); exit(0)
        default:
            FileHandle.standardError.write("unknown argument \(a)\n\(usage)\n".data(using: .utf8)!)
            exit(2)
        }
    }
    if o.maxGB > 8.0 { o.maxGB = 8.0 }
    if o.maxGB <= 0 { o.maxGB = 8.0 }
    if o.seconds <= 0 || o.seconds > 60 { o.seconds = 5.0 }
    if o.quick { o.seconds = 1.0; if o.shapes == nil { o.shapes = ["square"] } }
    return o
}

func say(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let opts = parseOptions()
if !opts.yes {
    say("metal4_bench uses the GPU at full load for several minutes and allocates up to \(Int(opts.maxGB)) GB.")
    say("Refusing to run without --yes.\n\(usage)")
    exit(2)
}
let capBytes = Int(opts.maxGB * 1024 * 1024 * 1024)

// MARK: - Device and libraries

guard let device = MTLCreateSystemDefaultDevice() else { say("no Metal device"); exit(1) }
guard let queue = device.makeCommandQueue() else { say("cannot create command queue"); exit(1) }

var deviceInfo: [String: Any] = [
    "name": device.name,
    "os": ProcessInfo.processInfo.operatingSystemVersionString,
    "recommended_max_working_set_gb": Double(device.recommendedMaxWorkingSetSize) / 1e9,
    "max_buffer_length_gb": Double(device.maxBufferLength) / 1e9,
]
if #available(macOS 14.0, *) { deviceInfo["architecture"] = device.architecture.name }
var isM5Class = false
if #available(macOS 26.0, *) {
    deviceInfo["supports_family_metal4"] = device.supportsFamily(.metal4)
    isM5Class = device.supportsFamily(.apple10)
    deviceInfo["supports_family_apple10"] = isM5Class
}
say("device: \(device.name)  os: \(ProcessInfo.processInfo.operatingSystemVersionString)")
if !isM5Class {
    say("WARNING: GPU does not report MTLGPUFamily.apple10 (M5 class). TensorOps then fall back to shader code; numbers say nothing about neural accelerators.")
}

func findShader() -> String? {
    var candidates: [String] = []
    if let p = opts.shaderPath { candidates.append(p) }
    let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
    candidates.append(exe.appendingPathComponent("kernels.metal").path)
    candidates.append(FileManager.default.currentDirectoryPath + "/kernels.metal")
    for c in candidates where FileManager.default.fileExists(atPath: c) { return c }
    return nil
}
guard let shaderPath = findShader(), let shaderSource = try? String(contentsOfFile: shaderPath, encoding: .utf8) else {
    say("cannot find kernels.metal (pass --shader PATH)"); exit(1)
}

var compileReport: [String: Any] = [:]

func compileLibrary(_ tag: String, define: String, metal4: Bool) -> MTLLibrary? {
    let o = MTLCompileOptions()
    if metal4 {
        if #available(macOS 26.0, *) {
            o.languageVersion = .version4_0
        } else {
            compileReport[tag] = "skipped: needs macOS 26 (MTLLanguageVersion.version4_0)"
            say("[\(tag)] skipped: needs macOS 26")
            return nil
        }
    } else {
        o.languageVersion = .version3_1
    }
    if #available(macOS 15.0, *) { o.mathMode = .safe }
    let t0 = Date()
    do {
        let lib = try device.makeLibrary(source: "#define \(define) 1\n" + shaderSource, options: o)
        compileReport[tag] = String(format: "ok (%.1f s)", Date().timeIntervalSince(t0))
        say("[\(tag)] compiled in \(String(format: "%.1f", Date().timeIntervalSince(t0))) s")
        return lib
    } catch {
        let msg = "\(error)"
        compileReport[tag] = "FAILED: " + String(msg.prefix(4000))
        say("[\(tag)] COMPILE FAILED:\n\(msg)")
        return nil
    }
}

let libSimd = compileLibrary("simdgroup_baseline", define: "BENCH_SIMD", metal4: false)
let libMPP = compileLibrary("tensor_ops_fp", define: "BENCH_MPP", metal4: true)
let libI8 = opts.int8 ? compileLibrary("tensor_ops_int8", define: "BENCH_MPP_INT8", metal4: true) : nil

var psoCache: [String: MTLComputePipelineState] = [:]
func pipeline(_ name: String) throws -> MTLComputePipelineState {
    if let p = psoCache[name] { return p }
    for lib in [libSimd, libMPP, libI8] {
        if let lib = lib, let f = lib.makeFunction(name: name) {
            let p = try device.makeComputePipelineState(function: f)
            psoCache[name] = p
            return p
        }
    }
    throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "function \(name) not found (its library failed to compile?)"])
}

// MARK: - Data types, params, buffers

enum DType: String {
    case fp16, bf16, int8
    var inSize: Int { self == .int8 ? 1 : 2 }
    var outSize: Int { self == .int8 ? 4 : 2 }
    var msl: String { switch self { case .fp16: return "half"; case .bf16: return "bfloat"; case .int8: return "i8" } }
}

struct GemmParams {
    var M: Int32, N: Int32, K: Int32, lda: Int32, ldb: Int32, ldc: Int32
    var tiles_m: Int32, tiles_n: Int32, swizzle: Int32, pad: Int32
}

struct XorShift {
    var s: UInt64
    mutating func next() -> UInt64 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s }
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) * 2 - 1 }  // [-1, 1)
}

@inline(__always) func bf16Bits(_ f: Float) -> UInt16 {
    let b = f.bitPattern
    return UInt16(truncatingIfNeeded: (b &+ 0x7FFF &+ ((b >> 16) & 1)) >> 16)
}
@inline(__always) func bf16Value(_ h: UInt16) -> Float { Float(bitPattern: UInt32(h) << 16) }

var allocated = 0
func makeBuffer(_ bytes: Int) throws -> MTLBuffer {
    if allocated + bytes > capBytes {
        throw NSError(domain: "bench", code: 2, userInfo: [NSLocalizedDescriptionKey:
            "allocation of \(bytes >> 20) MB would exceed the \(capBytes >> 30) GB cap"])
    }
    guard bytes <= device.maxBufferLength, let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
        throw NSError(domain: "bench", code: 3, userInfo: [NSLocalizedDescriptionKey: "makeBuffer(\(bytes)) failed"])
    }
    allocated += bytes
    return b
}
func release(_ bufs: [MTLBuffer]) { for b in bufs { allocated -= b.length } }

/// Fills a buffer with random values in [-1, 1) (int8: [-127, 127]); returns them as Float
/// when `keep` is set (for the CPU reference), rounded to the stored precision.
func fill(_ buf: MTLBuffer, count: Int, dtype: DType, seed: UInt64, keep: Bool) -> [Float] {
    var rng = XorShift(s: seed | 1)
    var kept: [Float] = keep ? [Float](repeating: 0, count: count) : []
    switch dtype {
    case .fp16:
        let p = buf.contents().bindMemory(to: UInt16.self, capacity: count)
        for i in 0..<count {
            let h = Float16(rng.unit())
            p[i] = h.bitPattern
            if keep { kept[i] = Float(h) }
        }
    case .bf16:
        let p = buf.contents().bindMemory(to: UInt16.self, capacity: count)
        for i in 0..<count {
            let b = bf16Bits(rng.unit())
            p[i] = b
            if keep { kept[i] = bf16Value(b) }
        }
    case .int8:
        let p = buf.contents().bindMemory(to: Int8.self, capacity: count)
        for i in 0..<count {
            let v = Int8(truncatingIfNeeded: Int(rng.next() % 255) - 127)
            p[i] = v
            if keep { kept[i] = Float(v) }
        }
    }
    return kept
}

func readOut(_ buf: MTLBuffer, count: Int, dtype: DType) -> [Float] {
    switch dtype {
    case .fp16:
        let p = buf.contents().bindMemory(to: UInt16.self, capacity: count)
        return (0..<count).map { Float(Float16(bitPattern: p[$0])) }
    case .bf16:
        let p = buf.contents().bindMemory(to: UInt16.self, capacity: count)
        return (0..<count).map { bf16Value(p[$0]) }
    case .int8:
        let p = buf.contents().bindMemory(to: Int32.self, capacity: count)
        return (0..<count).map { Float(p[$0]) }
    }
}

// MARK: - Variants

struct Variant {
    let label: String      // e.g. "c_tensorops_tuned sg32x32 w2x2 bk128 morton"
    let kind: String       // a_simdgroup | b_tensorops_basic | c_tensorops_tuned
    let function: String
    let tileM: Int
    let tileN: Int
    let threads: Int
    let bk: Int            // K must be a multiple of this
    let swizzle: Int       // 0 raster, 8 Morton in 8x8 blocks
}

let MORTON = 8

func variants(for dt: DType, K: Int) -> [Variant] {
    var v: [Variant] = []
    let bk = K % 128 == 0 ? 128 : (K % 64 == 0 ? 64 : 0)
    let configs: [(Int, Int, Int, Int)] = [(32, 32, 2, 2), (32, 32, 4, 4), (64, 64, 2, 2)]
    if dt == .int8 {
        guard K % 128 == 0 else { return [] }
        for (sm, sn, wm, wn) in configs {
            v.append(Variant(label: "c_tensorops_tuned int8 sg\(sm)x\(sn) w\(wm)x\(wn) bk128 morton",
                             kind: "c_tensorops_tuned", function: "mpp_i8_sg\(sm)x\(sn)_w\(wm)x\(wn)_bk128",
                             tileM: sm * wm, tileN: sn * wn, threads: 32 * wm * wn, bk: 128, swizzle: MORTON))
        }
        return v
    }
    v.append(Variant(label: "a_simdgroup 64x64x32 morton", kind: "a_simdgroup", function: "gemm_simd_\(dt.msl)",
                     tileM: 64, tileN: 64, threads: 128, bk: 32, swizzle: MORTON))
    v.append(Variant(label: "b_tensorops_basic simdgroups<4> 64x64 dynK raster", kind: "b_tensorops_basic",
                     function: "mpp_basic_\(dt.msl)", tileM: 64, tileN: 64, threads: 128, bk: 1, swizzle: 0))
    if bk > 0 {
        for (i, (sm, sn, wm, wn)) in configs.enumerated() {
            let fn = "mpp_tuned_\(dt.msl)_sg\(sm)x\(sn)_w\(wm)x\(wn)_bk\(bk)"
            v.append(Variant(label: "c_tensorops_tuned sg\(sm)x\(sn) w\(wm)x\(wn) bk\(bk) morton", kind: "c_tensorops_tuned",
                             function: fn, tileM: sm * wm, tileN: sn * wn, threads: 32 * wm * wn, bk: bk, swizzle: MORTON))
            if i == 0 {
                v.append(Variant(label: "c_tensorops_tuned sg\(sm)x\(sn) w\(wm)x\(wn) bk\(bk) raster", kind: "c_tensorops_tuned",
                                 function: fn, tileM: sm * wm, tileN: sn * wn, threads: 32 * wm * wn, bk: bk, swizzle: 0))
            }
        }
    }
    return v
}

func alignmentProblem(_ v: Variant, rows: Int, N: Int, K: Int) -> String? {
    if rows % v.tileM != 0 { return "rows \(rows) not a multiple of tile \(v.tileM)" }
    if N % v.tileN != 0 { return "N \(N) not a multiple of tile \(v.tileN)" }
    if K % v.bk != 0 { return "K \(K) not a multiple of \(v.bk)" }
    return nil
}

func params(_ v: Variant, rows: Int, N: Int, K: Int) -> (GemmParams, Int) {
    let tm = rows / v.tileM, tn = N / v.tileN
    let p = GemmParams(M: Int32(rows), N: Int32(N), K: Int32(K), lda: Int32(K), ldb: Int32(N), ldc: Int32(N),
                       tiles_m: Int32(tm), tiles_n: Int32(tn), swizzle: Int32(v.swizzle), pad: 0)
    let groups: Int
    if v.swizzle > 1 {
        let s = v.swizzle
        groups = ((tm + s - 1) / s * s) * ((tn + s - 1) / s * s)
    } else {
        groups = tm * tn
    }
    return (p, groups)
}

func encode(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, _ a: MTLBuffer, _ b: MTLBuffer, _ c: MTLBuffer,
            _ p: GemmParams, groups: Int, threads: Int, repeats: Int) {
    var pp = p
    enc.setComputePipelineState(pso)
    enc.setBuffer(a, offset: 0, index: 0)
    enc.setBuffer(b, offset: 0, index: 1)
    enc.setBuffer(c, offset: 0, index: 2)
    enc.setBytes(&pp, length: MemoryLayout<GemmParams>.stride, index: 3)
    for _ in 0..<repeats {
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }
}

func runOnce(_ v: Variant, _ a: MTLBuffer, _ b: MTLBuffer, _ c: MTLBuffer, rows: Int, N: Int, K: Int) throws {
    let pso = try pipeline(v.function)
    let (p, groups) = params(v, rows: rows, N: N, K: K)
    guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw NSError(domain: "bench", code: 4) }
    encode(enc, pso, a, b, c, p, groups: groups, threads: v.threads, repeats: 1)
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if let e = cb.error { throw e }
}

// MARK: - Verification (small cases, each variant vs CPU reference and vs the simdgroup baseline)

var verification: [[String: Any]] = []
var verifiedOK: [String: Bool] = [:]   // key: "<dtype>|<function>"

func verify(dt: DType, M: Int, N: Int, K: Int) {
    say("verify \(dt.rawValue) \(M)x\(N)x\(K)")
    do {
        let a = try makeBuffer(M * K * dt.inSize), b = try makeBuffer(K * N * dt.inSize)
        defer { release([a, b]) }
        let av = fill(a, count: M * K, dtype: dt, seed: 11, keep: true)
        let bv = fill(b, count: K * N, dtype: dt, seed: 23, keep: true)
        var ref = [Double](repeating: 0, count: M * N)
        for i in 0..<M { for k in 0..<K { let x = Double(av[i * K + k]); if x == 0 { continue }
            for j in 0..<N { ref[i * N + j] += x * Double(bv[k * N + j]) } } }
        let refMax = ref.map { abs($0) }.max() ?? 1
        // tolerance: output rounding of the stored type plus fp32-accumulation slack
        let (rtol, atol): (Double, Double) = dt == .fp16 ? (2e-3, 2e-3 * refMax) : (dt == .bf16 ? (1.6e-2, 1.6e-2 * refMax) : (0, 0))
        var baseline: [Float]? = nil
        for v in variants(for: dt, K: K) {
            var rec: [String: Any] = ["dtype": dt.rawValue, "shape": [M, N, K], "variant": v.label, "function": v.function]
            if let prob = alignmentProblem(v, rows: M, N: N, K: K) { rec["skipped"] = prob; verification.append(rec); continue }
            do {
                let c = try makeBuffer(M * N * dt.outSize)
                defer { release([c]) }
                memset(c.contents(), 0, c.length)
                try runOnce(v, a, b, c, rows: M, N: N, K: K)
                let out = readOut(c, count: M * N, dtype: dt)
                var maxAbs = 0.0, bad = 0
                for i in 0..<(M * N) {
                    let e = abs(Double(out[i]) - ref[i])
                    if !(e <= atol + rtol * abs(ref[i])) { bad += 1 }   // NaN counts as bad
                    if e > maxAbs || e.isNaN { maxAbs = e.isNaN ? .infinity : e }
                }
                rec["max_abs_err_vs_cpu"] = maxAbs.isFinite ? maxAbs : -1
                rec["ref_max_abs"] = refMax
                rec["mismatches"] = bad
                if v.kind == "a_simdgroup" { baseline = out }
                if let base = baseline, v.kind != "a_simdgroup" {
                    var d = 0.0
                    for i in 0..<(M * N) { d = max(d, Double(abs(out[i] - base[i]))) }
                    rec["max_abs_diff_vs_simdgroup"] = d.isFinite ? d : -1
                }
                let ok = bad == 0
                rec["ok"] = ok
                let key = "\(dt.rawValue)|\(v.function)"
                verifiedOK[key] = (verifiedOK[key] ?? true) && ok
                say("  \(ok ? "ok  " : "FAIL") \(v.label): max|err| \(String(format: "%.3g", maxAbs)), mismatches \(bad)")
            } catch {
                rec["error"] = "\(error.localizedDescription)"
                verifiedOK["\(dt.rawValue)|\(v.function)"] = false
                say("  ERROR \(v.label): \(error.localizedDescription)")
            }
            verification.append(rec)
        }
    } catch {
        verification.append(["dtype": dt.rawValue, "shape": [M, N, K], "error": "\(error.localizedDescription)"])
    }
}

// MARK: - Timing

struct Shape { let key: String; let name: String; let M: Int; let N: Int; let K: Int; let rowsPerDispatch: Int }

func timeVariant(_ v: Variant, dt: DType, shape s: Shape, a: MTLBuffer, b: MTLBuffer, c: MTLBuffer) throws -> [String: Any] {
    let pso = try pipeline(v.function)
    if pso.maxTotalThreadsPerThreadgroup < v.threads {
        throw NSError(domain: "bench", code: 5, userInfo: [NSLocalizedDescriptionKey:
            "pipeline allows only \(pso.maxTotalThreadsPerThreadgroup) threads per threadgroup, kernel needs \(v.threads) (register pressure?)"])
    }
    let (p, groups) = params(v, rows: s.rowsPerDispatch, N: s.N, K: s.K)
    let dispatchesPerIter = s.M / s.rowsPerDispatch
    let flopsPerIter = 2.0 * Double(s.M) * Double(s.N) * Double(s.K)

    func batch(_ iters: Int) throws -> Double {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw NSError(domain: "bench", code: 4) }
        encode(enc, pso, a, b, c, p, groups: groups, threads: v.threads, repeats: iters * dispatchesPerIter)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { throw e }
        return cb.gpuEndTime - cb.gpuStartTime
    }

    // Warm-up: at least 2 iterations and ~0.5 s, which also sizes the batch so that one
    // command buffer takes roughly 0.1-0.25 s (well below any GPU watchdog).
    var per = try batch(1)
    var warm = per
    while warm < 0.5 { per = try batch(1); warm += per }
    let perIter = max(per, 1e-6)
    let itersPerCB = max(1, min(4096, Int(0.15 / perIter)))

    var iters = 0, gpu = 0.0
    var best = Double.infinity
    let t0 = Date()
    while Date().timeIntervalSince(t0) < opts.seconds {
        let g = try batch(itersPerCB)
        gpu += g; iters += itersPerCB
        best = min(best, g / Double(itersPerCB))
    }
    let wall = Date().timeIntervalSince(t0)
    let gflops = flopsPerIter * Double(iters) / gpu / 1e9
    return [
        "gflops": gflops,
        "gflops_best_batch": flopsPerIter / best / 1e9,
        "ms_per_iter": gpu / Double(iters) * 1e3,
        "iters": iters,
        "gpu_seconds": gpu,
        "wall_seconds": wall,
        "dispatches_per_iter": dispatchesPerIter,
        "threadgroups_per_dispatch": groups,
    ]
}

// MARK: - Main

let mlxReferenceTflops: [String: Double] = ["bf16": 57.9, "fp16": 53.8, "fp32": 39.8]

var shapes: [Shape] = [
    Shape(key: "square", name: "square 4096x4096x4096", M: 4096, N: 4096, K: 4096, rowsPerDispatch: 4096),
    Shape(key: "qk", name: "attention QK^T 37440x128x37440", M: 37440, N: 37440, K: 128, rowsPerDispatch: opts.qkTile),
    Shape(key: "pv", name: "attention PV 37440x37440x128", M: 37440, N: 128, K: 37440, rowsPerDispatch: opts.pvTile),
]
if let want = opts.shapes { shapes = shapes.filter { want.contains($0.key) } }
for s in shapes where s.M % s.rowsPerDispatch != 0 || s.rowsPerDispatch % 64 != 0 {
    say("tile rows \(s.rowsPerDispatch) for \(s.key) must divide \(s.M) and be a multiple of 64"); exit(2)
}

let dtypes = opts.dtypes.compactMap { DType(rawValue: $0) }.filter { $0 != .int8 }

// Verification first.
for dt in dtypes {
    verify(dt: dt, M: 256, N: 256, K: 512)
    verify(dt: dt, M: 128, N: 192, K: 128)
}
if opts.int8 && libI8 != nil { verify(dt: .int8, M: 256, N: 256, K: 512) }

var results: [[String: Any]] = []
func runCases(dt: DType, shape s: Shape) {
    let aCount = s.rowsPerDispatch * s.K, bCount = s.K * s.N, cCount = s.rowsPerDispatch * s.N
    let need = aCount * dt.inSize + bCount * dt.inSize + cCount * dt.outSize
    say("== \(s.name) \(dt.rawValue): buffers \(need >> 20) MB (\(s.M / s.rowsPerDispatch) dispatch(es) per iteration)")
    var bufs: [MTLBuffer] = []
    do {
        let a = try makeBuffer(aCount * dt.inSize); bufs.append(a)
        let b = try makeBuffer(bCount * dt.inSize); bufs.append(b)
        let c = try makeBuffer(cCount * dt.outSize); bufs.append(c)
        _ = fill(a, count: aCount, dtype: dt, seed: 101, keep: false)
        _ = fill(b, count: bCount, dtype: dt, seed: 202, keep: false)
        for v in variants(for: dt, K: s.K) {
            var rec: [String: Any] = ["shape": s.name, "shape_key": s.key, "M": s.M, "N": s.N, "K": s.K,
                                      "rows_per_dispatch": s.rowsPerDispatch, "dtype": dt.rawValue,
                                      "variant": v.label, "kind": v.kind, "function": v.function]
            if let ok = verifiedOK["\(dt.rawValue)|\(v.function)"] { rec["verified"] = ok }
            if let prob = alignmentProblem(v, rows: s.rowsPerDispatch, N: s.N, K: s.K) {
                rec["skipped"] = prob; results.append(rec); say("  skip \(v.label): \(prob)"); continue
            }
            do {
                let r = try timeVariant(v, dt: dt, shape: s, a: a, b: b, c: c)
                rec.merge(r) { $1 }
                let g = r["gflops"] as! Double
                if let ref = mlxReferenceTflops[dt.rawValue] { rec["ratio_vs_mlx_4096_matmul"] = g / 1000 / ref }
                say(String(format: "  %8.0f GFLOP/s  %@", g, v.label))
            } catch {
                rec["error"] = "\(error.localizedDescription)"
                say("  ERROR \(v.label): \(error.localizedDescription)")
            }
            results.append(rec)
        }
    } catch {
        results.append(["shape": s.name, "dtype": dt.rawValue, "error": "\(error.localizedDescription)"])
        say("  ERROR: \(error.localizedDescription)")
    }
    release(bufs)
}

for s in shapes {
    for dt in dtypes { runCases(dt: dt, shape: s) }
}
if opts.int8 && libI8 != nil, let sq = shapes.first(where: { $0.key == "square" }) { runCases(dt: .int8, shape: sq) }

// Summary: best of each kind per (shape, dtype), and the speed-up of the best TensorOps
// variant over the simdgroup baseline.
var summary: [[String: Any]] = []
let groupsByKey = Dictionary(grouping: results.filter { $0["gflops"] != nil }) { "\($0["shape_key"]!)|\($0["dtype"]!)" }
for (key, rows) in groupsByKey.sorted(by: { $0.key < $1.key }) {
    var best: [String: (Double, String)] = [:]
    for r in rows {
        let k = r["kind"] as! String, g = r["gflops"] as! Double
        if g > (best[k]?.0 ?? 0) { best[k] = (g, r["variant"] as! String) }
    }
    var e: [String: Any] = ["shape_dtype": key]
    for (k, gv) in best { e["best_" + k] = ["gflops": gv.0, "variant": gv.1] as [String: Any] }
    let tensorBest = max(best["b_tensorops_basic"]?.0 ?? 0, best["c_tensorops_tuned"]?.0 ?? 0)
    if let a = best["a_simdgroup"]?.0, a > 0, tensorBest > 0 { e["speedup_best_tensorops_vs_simdgroup"] = tensorBest / a }
    let dt = String(key.split(separator: "|").last ?? "")
    if let ref = mlxReferenceTflops[dt], tensorBest > 0 { e["best_tensorops_vs_mlx_57_9_class_reference"] = tensorBest / 1000 / ref }
    summary.append(e)
}

let report: [String: Any] = [
    "tool": "shutter metal4_bench",
    "timestamp": ISO8601DateFormatter().string(from: Date()),
    "device": deviceInfo,
    "options": ["seconds_per_case": opts.seconds, "max_gb": opts.maxGB, "qk_tile_rows": opts.qkTile, "pv_tile_rows": opts.pvTile] as [String: Any],
    "compile": compileReport,
    "mlx_reference_tflops_4096_matmul": mlxReferenceTflops,
    "verification": verification,
    "results": results,
    "summary": summary,
    "notes": [
        "gflops = 2*M*N*K*iterations / summed GPU time (command buffer gpuStartTime..gpuEndTime)",
        "a_simdgroup = plain MSL simdgroup_matrix kernel (no neural accelerators); b/c = Metal 4 TensorOps (matmul2d)",
        "MLX >= 0.30 on macOS >= 26.2 already uses TensorOps (its 'NAX' kernels) for matmul and SDPA on M5: compare b/c with the MLX numbers, not only with a",
    ],
]

func sanitize(_ x: Any) -> Any {
    if let d = x as? Double { return d.isFinite ? d : -1 }
    if let a = x as? [Any] { return a.map(sanitize) }
    if let m = x as? [String: Any] { return m.mapValues(sanitize) }
    return x
}

do {
    let data = try JSONSerialization.data(withJSONObject: sanitize(report), options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    if let out = opts.out {
        let url = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
        say("wrote \(url.path)")
    }
} catch {
    say("JSON output failed: \(error)")
    exit(1)
}
