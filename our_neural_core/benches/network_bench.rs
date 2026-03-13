use criterion::{black_box, criterion_group, criterion_main, Criterion};
use our_neural_core::{Activation, Network};

fn bench_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");

    // Small network: 64 → 32 → 10
    {
        let net = Network::new(
            &[64, 32, 10],
            vec![Activation::ReLU, Activation::Sigmoid],
        );
        let input: Vec<f64> = (0..64).map(|i| (i as f64) * 0.01).collect();
        group.bench_function("small_64x32x10", |b| {
            b.iter(|| net.infer(black_box(&input)));
        });
    }

    // Medium network: 256 → 128 → 64 → 10
    {
        let net = Network::new(
            &[256, 128, 64, 10],
            vec![Activation::ReLU, Activation::ReLU, Activation::Sigmoid],
        );
        let input: Vec<f64> = (0..256).map(|i| (i as f64) * 0.001).collect();
        group.bench_function("medium_256x128x64x10", |b| {
            b.iter(|| net.infer(black_box(&input)));
        });
    }

    // MNIST-scale: 784 → 256 → 128 → 10
    {
        let net = Network::new(
            &[784, 256, 128, 10],
            vec![Activation::ReLU, Activation::ReLU, Activation::Sigmoid],
        );
        let input: Vec<f64> = (0..784).map(|i| (i as f64) * 0.001).collect();
        group.bench_function("mnist_scale_784x256x128x10", |b| {
            b.iter(|| net.infer(black_box(&input)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_inference);
criterion_main!(benches);
