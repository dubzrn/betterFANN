package configmesh_test

import (
	"testing"
	"time"

	"github.com/dubzrn/betterFANN/configmesh"
)

// ── 1. Basic set/get ──────────────────────────────────────────────────────────

func TestSetGet(t *testing.T) {
	mesh := configmesh.New()

	version, err := mesh.Set("model/learning_rate", "0.001")
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}
	if version != 1 {
		t.Errorf("expected version 1, got %d", version)
	}

	val, ok := mesh.Get("model/learning_rate")
	if !ok {
		t.Fatal("Get returned not-found for a key that was just Set")
	}
	if val != "0.001" {
		t.Errorf("expected '0.001', got %q", val)
	}
}

// ── 2. Version tracking ───────────────────────────────────────────────────────

func TestVersionIncrement(t *testing.T) {
	mesh := configmesh.New()

	v1, _ := mesh.Set("a", "first")
	v2, _ := mesh.Set("b", "second")
	v3, _ := mesh.Set("a", "updated")

	if v1 != 1 {
		t.Errorf("first write: expected version 1, got %d", v1)
	}
	if v2 != 2 {
		t.Errorf("second write: expected version 2, got %d", v2)
	}
	if v3 != 3 {
		t.Errorf("third write: expected version 3, got %d", v3)
	}

	gotV, ok := mesh.GetVersion("a")
	if !ok {
		t.Fatal("GetVersion returned not-found")
	}
	if gotV != 3 {
		t.Errorf("expected version 3 for key 'a', got %d", gotV)
	}
}

// ── 3. Commit index advances correctly ────────────────────────────────────────

func TestCommitIndexAdvances(t *testing.T) {
	mesh := configmesh.New()

	if mesh.CommitIndex() != 0 {
		t.Errorf("initial commit index should be 0, got %d", mesh.CommitIndex())
	}

	for i := 0; i < 5; i++ {
		if _, err := mesh.Set("k", "v"); err != nil {
			t.Fatal(err)
		}
	}

	if mesh.CommitIndex() != 5 {
		t.Errorf("expected commit index 5, got %d", mesh.CommitIndex())
	}
	if mesh.LogLen() != 5 {
		t.Errorf("expected log length 5, got %d", mesh.LogLen())
	}
}

// ── 4. Watch / notify ─────────────────────────────────────────────────────────

func TestWatchReceivesUpdateEvents(t *testing.T) {
	mesh := configmesh.New()
	ch := mesh.Watch("feature/flag", 4)
	defer mesh.Unwatch("feature/flag", ch)

	mesh.Set("feature/flag", "false") //nolint:errcheck
	mesh.Set("feature/flag", "true")  //nolint:errcheck

	// Drain up to 2 events with a short timeout.
	received := 0
	deadline := time.After(100 * time.Millisecond)
	for received < 2 {
		select {
		case ev, ok := <-ch:
			if !ok {
				t.Fatal("watch channel closed prematurely")
			}
			if ev.Key != "feature/flag" {
				t.Errorf("unexpected key in watch event: %q", ev.Key)
			}
			received++
		case <-deadline:
			t.Fatalf("timed out waiting for watch events; received %d/2", received)
		}
	}

	if received != 2 {
		t.Errorf("expected 2 watch events, got %d", received)
	}
}

// ── 5. Missing key returns false ──────────────────────────────────────────────

func TestGetMissingKeyReturnsFalse(t *testing.T) {
	mesh := configmesh.New()
	_, ok := mesh.Get("does-not-exist")
	if ok {
		t.Error("Get must return false for a key that has never been Set")
	}
}
