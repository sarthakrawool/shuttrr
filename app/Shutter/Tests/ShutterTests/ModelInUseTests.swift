import XCTest
@testable import Shutter

final class ModelInUseTests: XCTestCase {
    private func decode(_ json: String) throws -> ModelInfo {
        try JSONDecoder().decode(ModelInfo.self, from: Data(json.utf8))
    }

    func testInUseStreamedReadsAsInUseFromSSD() throws {
        let m = try decode(#"{"id":"z","state":"in_use","in_use":{"request":"abc","rung":"stream_tight","residency":"streamed"}}"#)
        XCTAssertTrue(m.isInUse)
        XCTAssertFalse(m.isLoaded)
        XCTAssertEqual(m.inUseRung, "stream_tight")
        XCTAssertEqual(m.inUseText, "In use by a request · streams from SSD")
    }

    func testInUseWithoutStepAndOlderEngines() throws {
        let m = try decode(#"{"id":"z","state":"in_use","in_use":{"request":"abc","rung":null,"residency":null}}"#)
        XCTAssertEqual(m.inUseText, "In use by a request")
        let old = try decode(#"{"id":"z","state":"loaded","in_use":null}"#)
        XCTAssertFalse(old.isInUse)
        XCTAssertTrue(old.isLoaded)
    }
}
