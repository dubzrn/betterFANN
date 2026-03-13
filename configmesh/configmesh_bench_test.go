package configmesh_test

import (
	"fmt"
	"testing"

	"github.com/dubzrn/betterFANN/configmesh"
)

// BenchmarkSet measures the throughput of Set operations on a fresh ConfigMesh.
func BenchmarkSet(b *testing.B) {
	mesh := configmesh.New()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("bench/key/%d", i%100)
		if _, err := mesh.Set(key, "value"); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGet measures the throughput of Get operations after pre-loading keys.
func BenchmarkGet(b *testing.B) {
	mesh := configmesh.New()
	const numKeys = 100
	for i := 0; i < numKeys; i++ {
		mesh.Set(fmt.Sprintf("bench/key/%d", i), fmt.Sprintf("val%d", i)) //nolint:errcheck
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("bench/key/%d", i%numKeys)
		mesh.Get(key)
	}
}

// BenchmarkSetGet measures an interleaved Set/Get workload (write-then-read).
func BenchmarkSetGet(b *testing.B) {
	mesh := configmesh.New()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("bench/key/%d", i%50)
		mesh.Set(key, "v") //nolint:errcheck
		mesh.Get(key)
	}
}

// BenchmarkWatch measures the cost of registering and deregistering watchers.
func BenchmarkWatch(b *testing.B) {
	mesh := configmesh.New()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := fmt.Sprintf("bench/watch/%d", i%20)
		ch := mesh.Watch(key, 1)
		mesh.Unwatch(key, ch)
	}
}
