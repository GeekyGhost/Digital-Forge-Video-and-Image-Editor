use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Rational {
    pub numerator: i64,
    pub denominator: i64,
}

impl Rational {
    pub fn validate(self) -> Result<Self, String> {
        if self.denominator <= 0 {
            return Err("rational denominator must be positive".into());
        }
        Ok(self)
    }

    pub fn as_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectEnvelope {
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    pub updated_at: String,
    #[serde(flatten)]
    pub remainder: serde_json::Map<String, serde_json::Value>,
}

impl ProjectEnvelope {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!("unsupported project schema {}", self.schema_version));
        }
        if self.id.trim().is_empty() || self.name.trim().is_empty() {
            return Err("project id and name are required".into());
        }
        Ok(())
    }
}
