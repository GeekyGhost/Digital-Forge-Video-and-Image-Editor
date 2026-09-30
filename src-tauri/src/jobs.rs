use parking_lot::RwLock;
use serde::Serialize;
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeJob {
    pub id: Uuid,
    pub kind: String,
    pub state: String,
    pub progress: f32,
    pub message: String,
}

#[derive(Default, Clone)]
pub struct JobRegistry(Arc<RwLock<HashMap<Uuid, NativeJob>>>);

impl JobRegistry {
    pub fn queue(&self, kind: String) -> NativeJob {
        let job = NativeJob {
            id: Uuid::new_v4(),
            kind,
            state: "queued".into(),
            progress: 0.0,
            message: "Waiting for a model lease".into(),
        };
        self.0.write().insert(job.id, job.clone());
        job
    }

    pub fn list(&self) -> Vec<NativeJob> {
        self.0.read().values().cloned().collect()
    }
}

