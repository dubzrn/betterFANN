use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use eloptic_classifier::{
    weights::SecureWeights, ElopticClassifier, ReLU, Sigmoid,
};
use std::sync::Arc;

// ── Dot-product benchmarks ──────────────────────────────────────────────────

/// Naive scalar dot product used as a baseline for comparison.
fn naive_dot(w: &[f64], input: &[f64]) -> f64 {
    w.iter().zip(input).map(|(a, b)| a * b).sum()
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for &n in &[64_usize, 256, 1024, 4096] {
        let weights: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
        let input: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
        let sw = SecureWeights::from_vec(weights.clone());

        group.bench_with_input(BenchmarkId::new("unrolled_4x", n), &n, |b, _| {
            b.iter(|| sw.dot(black_box(&input)));
        });

        group.bench_with_input(BenchmarkId::new("naive_scalar", n), &n, |b, _| {
            b.iter(|| naive_dot(black_box(&weights), black_box(&input)));
        });
    }
    group.finish();
}

// ── Forward-pass benchmarks ──────────────────────────────────────────────────

fn bench_forward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_pass");

    // Small network: 64 → 32 → 16
    {
        let clf = ElopticClassifier::new(vec![
            (64, 32, Arc::new(ReLU) as Arc<dyn eloptic_classifier::Activation>),
            (32, 16, Arc::new(Sigmoid)),
        ]);
        let input: Vec<f64> = (0..64).map(|i| (i as f64) * 0.01).collect();
        group.bench_function("small_64x32x16", |b| {
            b.iter(|| clf.forward(black_box(&input)));
        });
    }

    // Medium network: 256 → 128 → 64 → 10
    {
        let clf = ElopticClassifier::new(vec![
            (256, 128, Arc::new(ReLU) as Arc<dyn eloptic_classifier::Activation>),
            (128, 64, Arc::new(ReLU)),
            (64, 10, Arc::new(Sigmoid)),
        ]);
        let input: Vec<f64> = (0..256).map(|i| (i as f64) * 0.001).collect();
        group.bench_function("medium_256x128x64x10", |b| {
            b.iter(|| clf.forward(black_box(&input)));
        });
    }

    // Large network: 784 → 512 → 256 → 10  (MNIST-scale)
    {
        let clf = ElopticClassifier::new(vec![
            (784, 512, Arc::new(ReLU) as Arc<dyn eloptic_classifier::Activation>),
            (512, 256, Arc::new(ReLU)),
            (256, 10, Arc::new(Sigmoid)),
        ]);
        let input: Vec<f64> = (0..784).map(|i| (i as f64) * 0.001).collect();
        group.bench_function("large_784x512x256x10", |b| {
            b.iter(|| clf.forward(black_box(&input)));
        });
    }

    group.finish();
}

// ── Training-step benchmark ──────────────────────────────────────────────────

fn bench_train_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("train_step");

    {
        let mut clf = ElopticClassifier::new(vec![
            (64, 32, Arc::new(ReLU) as Arc<dyn eloptic_classifier::Activation>),
            (32, 10, Arc::new(Sigmoid)),
        ]);
        let input: Vec<f64> = (0..64).map(|i| (i as f64) * 0.01).collect();
        let target: Vec<f64> = vec![0.0; 10];
        group.bench_function("small_64x32x10", |b| {
            b.iter(|| clf.train_step(black_box(&input), black_box(&target), 0.01));
        });
    }

    {
        let mut clf = ElopticClassifier::new(vec![
            (256, 128, Arc::new(ReLU) as Arc<dyn eloptic_classifier::Activation>),
            (128, 10, Arc::new(Sigmoid)),
        ]);
        let input: Vec<f64> = (0..256).map(|i| (i as f64) * 0.001).collect();
        let target: Vec<f64> = vec![0.0; 10];
        group.bench_function("medium_256x128x10", |b| {
            b.iter(|| clf.train_step(black_box(&input), black_box(&target), 0.01));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dot_product,
    bench_forward_pass,
    bench_train_step
);
criterion_main!(benches);
