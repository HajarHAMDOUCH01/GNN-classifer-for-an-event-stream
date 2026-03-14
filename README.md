# SO(n) Conformance alignement for Petri Nets

## Approach

Embeds the reachability graph of a Petri net into SO(n):

- Each reachable marking is assigned an orthogonal axis of an n-dimensional sphere
- Each transition is encoded as an antisymmetric Lie algebra generator Omega_t in so(n)
- Firing a transition corresponds to applying the rotation R_t = exp(Omega_t)

A neural network is trained to navigate this geometric space: given a source marking
v_src and a target marking v_tgt (both unit vectors on the sphere), it produces the
sequence of conformant transitions that recovers the process instance back to a
conformant state.

## Online Conformance Checking Use Case

Given an event stream t1 -> t2 -> t_nonconformant -> ...:

1. Track v_current after each conformant transition fires
2. When a non-conformant transition t_x arrives:
   - Find the marking in the reachability graph where t_x is enabled — this is v_tgt
   - Query model(v_src=v_current, v_tgt=found_marking)
   - Execute the returned recovery path, then fire t_x
3. Continue tracking from the updated marking

## Current Status

Proven on fixed graphs :

- The SO(n) encoding correctly represents any bounded Petri net reachability graph
- The combined loss (cross-entropy + state equation constraint) trains successfully
- The model learns to navigate the embedded graph and recover conformant paths

Current limitation: the reachability tensor (omegas) is registered as a fixed model
buffer, tying each trained instance to one specific Petri net. Training and evaluation
are performed on the same graph, so generalization across nets has not yet been
demonstrated.

## Next Step

Decouple omegas from the model: pass the reachability tensor as a runtime input.
Reformulate the model input as graph-invariant scalar features derived from the SO(n)
geometry — specifically, inner products between candidate next positions and v_tgt —
enabling a single trained model to navigate any Petri net encoded in this framework
without retraining. This would constitute zero-shot transfer across process models,
which is the core scientific claim this architecture is designed to support.