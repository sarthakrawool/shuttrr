//! In-process event bus (port of `queue.EventBus`): fan-out to any number of subscribers
//! (SSE streams), each with a bounded queue that drops its oldest events when slow, plus a
//! history of recent events for new subscribers.

use crate::util::now;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

#[derive(Debug, Clone)]
pub struct Event {
    pub seq: u64,
    pub kind: String,
    pub job_id: Option<String>,
    pub data: Value,
    pub ts: f64,
}

impl Event {
    pub fn to_json(&self) -> Value {
        json!({"kind": self.kind, "job_id": self.job_id, "data": self.data, "ts": self.ts})
    }
}

struct Inner {
    recent: VecDeque<Arc<Event>>,
    seq: u64,
}

pub struct EventBus {
    tx: broadcast::Sender<Arc<Event>>,
    inner: Mutex<Inner>,
    maxlen: usize,
}

impl EventBus {
    pub fn new(maxlen: usize) -> Self {
        let (tx, _) = broadcast::channel(500);
        EventBus { tx, inner: Mutex::new(Inner { recent: VecDeque::with_capacity(maxlen), seq: 0 }), maxlen }
    }

    pub fn publish(&self, kind: &str, job_id: Option<&str>, data: Value) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.seq += 1;
        let ev = Arc::new(Event { seq: g.seq, kind: kind.into(), job_id: job_id.map(String::from), data, ts: now() });
        if g.recent.len() == self.maxlen {
            g.recent.pop_front();
        }
        g.recent.push_back(ev.clone());
        // send under the lock so live order == history order
        let _ = self.tx.send(ev);
    }

    /// Subscribe and get the last `n` events (oldest first) atomically: every later event
    /// arrives on the receiver exactly once and none of the returned ones do.
    pub fn subscribe_with_recent(&self, n: usize) -> (broadcast::Receiver<Arc<Event>>, Vec<Arc<Event>>) {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let rx = self.tx.subscribe();
        let skip = g.recent.len().saturating_sub(n);
        (rx, g.recent.iter().skip(skip).cloned().collect())
    }

    pub fn recent(&self, n: usize) -> Vec<Arc<Event>> {
        self.subscribe_with_recent(n).1
    }

    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn recent_then_live_no_duplicates() {
        let bus = EventBus::new(3);
        for i in 0..5 {
            bus.publish("x", None, json!({"i": i}));
        }
        let (mut rx, recent) = bus.subscribe_with_recent(2);
        assert_eq!(recent.iter().map(|e| e.data["i"].as_i64().unwrap()).collect::<Vec<_>>(), vec![3, 4]);
        assert_eq!(bus.recent(10).len(), 3, "history is bounded");
        bus.publish("y", Some("j"), json!({}));
        let ev = rx.recv().await.unwrap();
        assert_eq!(ev.kind, "y");
        assert_eq!(ev.to_json()["job_id"], "j");
        assert_eq!(bus.subscriber_count(), 1);
    }
}
