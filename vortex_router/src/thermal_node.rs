/// A node in the vortex pool characterised by its thermal state.
#[derive(Debug, Clone)]
pub struct ThermalNode {
    pub id: u64,
    pub temperature: f64,
    pub latency_ms: f64,
    pub quality_improving: bool,
}

impl ThermalNode {
    pub fn new(id: u64, temperature: f64, latency_ms: f64) -> Self {
        Self { id, temperature, latency_ms, quality_improving: false }
    }

    /// Record a completed request; cooling the node and marking quality as improving.
    pub fn record_completion(&mut self, latency_ms: f64) {
        self.latency_ms = latency_ms;
        // A lower temperature than before means the node is cooling → improving.
        let new_temp = (self.temperature - 0.5_f64).max(0.0);
        self.quality_improving = new_temp < self.temperature;
        self.temperature = new_temp;
    }

    /// Record a failure; heat the node and mark quality as degrading.
    pub fn record_failure(&mut self) {
        self.temperature += 2.0;
        self.quality_improving = false;
    }

    /// Thermal score used for sorting: lower is colder/better.
    pub fn thermal_score(&self) -> f64 {
        self.temperature + self.latency_ms / 1000.0
    }
}
