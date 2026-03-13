use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sphere_node::crdt::{WeightCell, WeightSet};

fn bench_crdt_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("crdt_merge");

    // Single WeightCell merge — fast path
    group.bench_function("single_cell_merge", |b| {
        let mut base = WeightCell { value: 0.5, clock: 1, origin: 1 };
        let incoming = WeightCell { value: 0.9, clock: 5, origin: 2 };
        b.iter(|| {
            base.merge(black_box(&incoming));
        });
    });

    // WeightSet merge at various sizes
    for &n in &[16_usize, 64, 256, 1024] {
        let mut base_set = WeightSet::new(n, 1);
        let vals_a: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
        base_set.write_all(&vals_a);

        let mut incoming_set = WeightSet::new(n, 2);
        let vals_b: Vec<f64> = (0..n).map(|i| i as f64 * 0.002).collect();
        incoming_set.write_all(&vals_b);
        incoming_set.write_all(&vals_b); // clock = 2 (beats base_set clock = 1)

        group.bench_with_input(BenchmarkId::new("weight_set", n), &n, |b, _| {
            b.iter(|| {
                let mut s = base_set.clone();
                s.merge(black_box(&incoming_set));
                s
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_crdt_merge);
criterion_main!(benches);
