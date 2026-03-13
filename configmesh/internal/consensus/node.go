// Package consensus implements a single-node Raft-like consensus log for
// configuration entries.
//
// Design:
//
//   - A Node maintains an append-only log of LogEntry values.
//   - Each entry carries a term, an index, and a key/value pair.
//   - Commit advances the commitIndex, making entries visible to readers.
//   - Entries are applied to an in-memory state machine (the config store).
//
// This is a deterministic, synchronous implementation suitable for a single
// leader node.  A production deployment would replace the in-process log with
// a network-replicated Raft library (e.g. etcd/raft).
package consensus

import (
	"errors"
	"sync"
)

// ErrIndexOutOfRange is returned when an entry index is outside the log.
var ErrIndexOutOfRange = errors.New("consensus: index out of range")

// LogEntry is one entry in the consensus log.
type LogEntry struct {
	Term  uint64
	Index uint64
	Key   string
	Value string
}

// StateMachine is the interface that a consensus node drives on commit.
type StateMachine interface {
	Apply(entry LogEntry)
}

// Node is a single-leader consensus log node.
type Node struct {
	mu          sync.RWMutex
	log         []LogEntry
	commitIndex uint64 // last committed index (1-based; 0 = nothing committed)
	term        uint64
	sm          StateMachine
}

// NewNode constructs a Node that applies committed entries to sm.
func NewNode(sm StateMachine) *Node {
	return &Node{sm: sm}
}

// CurrentTerm returns the node's current Raft term.
func (n *Node) CurrentTerm() uint64 {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.term
}

// Propose appends a new entry to the log in the current term.
// The entry is NOT yet committed; call Commit to advance commitIndex.
func (n *Node) Propose(key, value string) LogEntry {
	n.mu.Lock()
	defer n.mu.Unlock()
	idx := uint64(len(n.log) + 1)
	entry := LogEntry{
		Term:  n.term,
		Index: idx,
		Key:   key,
		Value: value,
	}
	n.log = append(n.log, entry)
	return entry
}

// Commit advances the commitIndex to upTo (inclusive) and applies all newly
// committed entries to the state machine.
//
// Returns ErrIndexOutOfRange if upTo exceeds the length of the log.
func (n *Node) Commit(upTo uint64) error {
	n.mu.Lock()
	defer n.mu.Unlock()

	logLen := uint64(len(n.log))
	if upTo > logLen {
		return ErrIndexOutOfRange
	}
	for i := n.commitIndex; i < upTo; i++ {
		n.sm.Apply(n.log[i])
	}
	n.commitIndex = upTo
	return nil
}

// CommitIndex returns the current commit frontier.
func (n *Node) CommitIndex() uint64 {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.commitIndex
}

// LogLen returns the number of entries in the log.
func (n *Node) LogLen() int {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return len(n.log)
}

// IncrementTerm simulates a leader election that bumps the term counter.
func (n *Node) IncrementTerm() uint64 {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.term++
	return n.term
}

// EntryAt returns the log entry at 1-based index idx.
func (n *Node) EntryAt(idx uint64) (LogEntry, error) {
	n.mu.RLock()
	defer n.mu.RUnlock()
	if idx == 0 || int(idx) > len(n.log) {
		return LogEntry{}, ErrIndexOutOfRange
	}
	return n.log[idx-1], nil
}
