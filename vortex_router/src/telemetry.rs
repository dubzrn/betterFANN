/// Concentrated pulse telemetry emitted at batch completion — not a continuous bleed.
#[derive(Debug, Default, Clone)]
pub struct EnergyCannonTelemetry {
    pub batch_id: u64,
    pub total_requests: u64,
    pub failed_requests: u64,
    pub avg_latency_ms: f64,
    pub avg_temperature: f64,
}

/// Accumulates per-request observations and emits a single telemetry pulse per batch.
#[derive(Debug, Default)]
pub struct TelemetryAccumulator {
    batch_id: u64,
    latencies: Vec<f64>,
    temperatures: Vec<f64>,
    failures: u64,
}

impl TelemetryAccumulator {
    pub fn record(&mut self, latency_ms: f64, temperature: f64, success: bool) {
        self.latencies.push(latency_ms);
        self.temperatures.push(temperature);
        if !success {
            self.failures += 1;
        }
    }

    /// Drain accumulated observations and emit one telemetry pulse.
    pub fn emit(&mut self) -> EnergyCannonTelemetry {
        let n = self.latencies.len() as f64;
        let avg_latency = if n > 0.0 { self.latencies.iter().sum::<f64>() / n } else { 0.0 };
        let avg_temp = if n > 0.0 { self.temperatures.iter().sum::<f64>() / n } else { 0.0 };
        let pulse = EnergyCannonTelemetry {
            batch_id: self.batch_id,
            total_requests: self.latencies.len() as u64,
            failed_requests: self.failures,
            avg_latency_ms: avg_latency,
            avg_temperature: avg_temp,
        };
        self.batch_id += 1;
        self.latencies.clear();
        self.temperatures.clear();
        self.failures = 0;
        pulse
    }
}
