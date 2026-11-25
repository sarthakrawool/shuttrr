//! shutterd: the always-on part of Shutter. HTTP API, job store, queue, memory guard,
//! GPU locks, events and the watchdog; the model code runs in a Python worker child.

pub mod api;
pub mod cli;
pub mod config;
pub mod engine;
pub mod events;
pub mod files;
pub mod httpc;
pub mod images;
pub mod locks;
pub mod memory;
pub mod modes;
pub mod protocol;
pub mod stats;
pub mod store;
pub mod util;
pub mod worker;
