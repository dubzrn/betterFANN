// Package configmesh provides distributed configuration management with
// consensus guarantees.
//
// Usage:
//
//	mesh := configmesh.New()
//	mesh.Set("feature/dark-mode", "true")
//	val, _ := mesh.Get("feature/dark-mode")
//	fmt.Println(val) // "true"
package configmesh

import (
	"github.com/dubzrn/betterFANN/configmesh/internal/consensus"
	"github.com/dubzrn/betterFANN/configmesh/internal/store"
)

// ConfigMesh is the public API for distributed configuration management.
//
// Internally it chains a consensus.Node (append-only log with commit
// semantics) and a store.Store (versioned key-value state machine).
type ConfigMesh struct {
	node  *consensus.Node
	store *store.Store
}

// New constructs a ConfigMesh with an empty log and an empty store.
func New() *ConfigMesh {
	s := store.New()
	n := consensus.NewNode(s)
	return &ConfigMesh{node: n, store: s}
}

// Set proposes and immediately commits a key-value configuration entry.
//
// Returns the log index (version) at which the entry was committed.
func (m *ConfigMesh) Set(key, value string) (uint64, error) {
	entry := m.node.Propose(key, value)
	if err := m.node.Commit(entry.Index); err != nil {
		return 0, err
	}
	return entry.Index, nil
}

// Get returns the current value for key.
// The second return value is false when the key has not been set.
func (m *ConfigMesh) Get(key string) (string, bool) {
	cv, ok := m.store.Get(key)
	if !ok {
		return "", false
	}
	return cv.Value, true
}

// GetVersion returns the version (log index) at which key was last written.
func (m *ConfigMesh) GetVersion(key string) (uint64, bool) {
	cv, ok := m.store.Get(key)
	if !ok {
		return 0, false
	}
	return cv.Version, true
}

// Watch returns a buffered channel that receives a store.WatchEvent each time
// key is updated.
func (m *ConfigMesh) Watch(key string, bufSize int) chan store.WatchEvent {
	return m.store.Watch(key, bufSize)
}

// Unwatch closes and removes a watch subscription.
func (m *ConfigMesh) Unwatch(key string, ch chan store.WatchEvent) {
	m.store.Unwatch(key, ch)
}

// CommitIndex returns the highest committed log index.
func (m *ConfigMesh) CommitIndex() uint64 {
	return m.node.CommitIndex()
}

// LogLen returns the total number of proposed entries (committed or not).
func (m *ConfigMesh) LogLen() int {
	return m.node.LogLen()
}
