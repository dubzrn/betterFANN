// Package store implements a versioned, watchable configuration key-value store
// driven by the consensus log.
package store

import (
	"fmt"
	"sync"

	"github.com/dubzrn/betterFANN/configmesh/internal/consensus"
)

// ConfigVersion bundles a value with the consensus log index at which it was
// written.
type ConfigVersion struct {
	Value   string
	Version uint64 // log index that produced this value
}

// WatchEvent is emitted on a watch channel whenever a key is updated.
type WatchEvent struct {
	Key        string
	OldVersion ConfigVersion
	NewVersion ConfigVersion
}

// Store is a versioned key-value config store backed by the consensus log.
//
// It implements consensus.StateMachine so the consensus.Node can apply
// committed entries directly.
type Store struct {
	mu       sync.RWMutex
	data     map[string]ConfigVersion
	watchers map[string][]chan WatchEvent
}

// New creates an empty Store.
func New() *Store {
	return &Store{
		data:     make(map[string]ConfigVersion),
		watchers: make(map[string][]chan WatchEvent),
	}
}

// Apply implements consensus.StateMachine — called by the consensus node on
// commit.
func (s *Store) Apply(entry consensus.LogEntry) {
	s.mu.Lock()
	old := s.data[entry.Key]
	nv := ConfigVersion{Value: entry.Value, Version: entry.Index}
	s.data[entry.Key] = nv
	watchers := s.watchers[entry.Key]
	s.mu.Unlock()

	// Notify watchers outside the lock to avoid potential deadlock.
	for _, ch := range watchers {
		ev := WatchEvent{Key: entry.Key, OldVersion: old, NewVersion: nv}
		select {
		case ch <- ev:
		default:
			// Drop if the watcher is not consuming fast enough.
		}
	}
}

// Get returns the current value and version for key.
// The second return value is false when the key does not exist.
func (s *Store) Get(key string) (ConfigVersion, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cv, ok := s.data[key]
	return cv, ok
}

// GetAt returns the value if its version equals exactly wantVersion.
// Use this for read-your-writes consistency assertions.
func (s *Store) GetAt(key string, wantVersion uint64) (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cv, ok := s.data[key]
	if !ok {
		return "", fmt.Errorf("store: key %q not found", key)
	}
	if cv.Version != wantVersion {
		return "", fmt.Errorf(
			"store: key %q at version %d, want %d",
			key, cv.Version, wantVersion,
		)
	}
	return cv.Value, nil
}

// Watch subscribes to updates for key.  Returns a buffered channel that
// receives WatchEvent values.  Call Unwatch to close the subscription.
func (s *Store) Watch(key string, bufSize int) chan WatchEvent {
	ch := make(chan WatchEvent, bufSize)
	s.mu.Lock()
	s.watchers[key] = append(s.watchers[key], ch)
	s.mu.Unlock()
	return ch
}

// Unwatch removes and closes a watch channel.
func (s *Store) Unwatch(key string, ch chan WatchEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ws := s.watchers[key]
	for i, w := range ws {
		if w == ch {
			s.watchers[key] = append(ws[:i], ws[i+1:]...)
			close(ch)
			return
		}
	}
}

// Keys returns a snapshot of all keys currently in the store.
func (s *Store) Keys() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	keys := make([]string, 0, len(s.data))
	for k := range s.data {
		keys = append(keys, k)
	}
	return keys
}
