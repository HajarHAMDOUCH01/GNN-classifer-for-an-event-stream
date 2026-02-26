# """this example is N1 Petri Net from paper : relating event streams to process
# models using prefix-alignments"""

# from snakes.nets import *

# n = PetriNet("N1_example_1")

# # Places
# n.add_place(Place("p0", [0]))
# n.add_place(Place("p1", []))
# n.add_place(Place("p2", []))
# n.add_place(Place("p3", []))
# n.add_place(Place("p4", []))
# n.add_place(Place("p5", []))
# n.add_place(Place("p6", []))

# # Transitions 
# n.add_transition(Transition("t1"))
# n.add_transition(Transition("t2"))
# n.add_transition(Transition("t3"))
# n.add_transition(Transition("t4"))
# n.add_transition(Transition("t5"))
# n.add_transition(Transition("t6"))
# n.add_transition(Transition("t7"))
# n.add_transition(Transition("t8"))

# # Inputs
# n.add_input("p0", "t1", Variable("x"))
# n.add_input("p1", "t2", Variable("x"))
# n.add_input("p2", "t3", Variable("x"))
# n.add_input("p4", "t4", Variable("x"))
# n.add_input("p1", "t4", Variable("x"))
# n.add_input("p3", "t5", Variable("x"))
# n.add_input("p4", "t5", Variable("x"))
# n.add_input("p5", "t6", Variable("x"))
# n.add_input("p5", "t7", Variable("x"))
# n.add_input("p5", "t8", Variable("x"))

# # Outputs
# n.add_output("p1", "t1", Variable("x"))
# n.add_output("p2", "t1", Variable("x"))
# n.add_output("p3", "t2", Variable("x"))
# n.add_output("p4", "t3", Variable("x"))
# n.add_output("p5", "t4", Variable("x"))
# n.add_output("p5", "t5", Variable("x"))
# n.add_output("p1", "t6", Variable("x"))
# n.add_output("p2", "t6", Variable("x"))
# n.add_output("p6", "t7", Variable("x"))
# n.add_output("p6", "t8", Variable("x"))



# """
# === Reachability Graph ===

# From: {p0={0}}
#   --[t1]--> {p1={0}, p2={0}}

# From: {p1={0}, p2={0}}
#   --[t2]--> {p2={0}, p3={0}}
#   --[t3]--> {p1={0}, p4={0}}

# From: {p2={0}, p3={0}}
#   --[t3]--> {p3={0}, p4={0}}

# From: {p1={0}, p4={0}}
#   --[t2]--> {p3={0}, p4={0}}
#   --[t4]--> {p5={0}}

# From: {p3={0}, p4={0}}
#   --[t5]--> {p5={0}}

# From: {p5={0}}
#   --[t6]--> {p1={0}, p2={0}}
#   --[t7]--> {p6={0}}
#   --[t8]--> {p6={0}}

# From: {p6={0}}

# Total states: 7
# """




from snakes.nets import *

n = PetriNet("peer_review")

# Places (initial marking: p0=1, all others=0)
n.add_place(Place("p0",  [0]))
n.add_place(Place("p7",  []))
n.add_place(Place("p8",  []))
n.add_place(Place("p9",  []))
n.add_place(Place("p12", []))
n.add_place(Place("p16", []))
n.add_place(Place("p17", []))
n.add_place(Place("p18", []))
n.add_place(Place("p19", []))
n.add_place(Place("p20", []))
n.add_place(Place("p22", []))
n.add_place(Place("p23", []))

# Transitions
n.add_transition(Transition("t0"))
n.add_transition(Transition("t2"))
n.add_transition(Transition("t3"))
n.add_transition(Transition("t4"))
n.add_transition(Transition("t5"))
n.add_transition(Transition("t6"))
n.add_transition(Transition("t7"))
n.add_transition(Transition("t8"))
n.add_transition(Transition("t9"))
n.add_transition(Transition("t10"))
n.add_transition(Transition("t11"))
n.add_transition(Transition("t12"))
n.add_transition(Transition("t13"))
n.add_transition(Transition("t14"))

# Inputs (place -> transition)
n.add_input("p0",  "t0",  Variable("x"))

n.add_input("p16", "t6",  Variable("x"))
n.add_input("p16", "t7",  Variable("x"))

n.add_input("p7",  "t8",  Variable("x"))

n.add_input("p19", "t2",  Variable("x"))
n.add_input("p19", "t3",  Variable("x"))

n.add_input("p8",  "t8",  Variable("x"))

n.add_input("p20", "t4",  Variable("x"))
n.add_input("p20", "t5",  Variable("x"))

n.add_input("p9",  "t8",  Variable("x"))

n.add_input("p12", "t9",  Variable("x"))

n.add_input("p17", "t10", Variable("x"))
n.add_input("p17", "t11", Variable("x"))

n.add_input("p18", "t12", Variable("x"))
n.add_input("p18", "t13", Variable("x"))
n.add_input("p18", "t14", Variable("x"))

# Outputs (transition -> place)
n.add_output("p16", "t0",  Variable("x"))
n.add_output("p19", "t0",  Variable("x"))
n.add_output("p20", "t0",  Variable("x"))

n.add_output("p7",  "t6",  Variable("x"))
n.add_output("p7",  "t7",  Variable("x"))

n.add_output("p8",  "t2",  Variable("x"))
n.add_output("p8",  "t3",  Variable("x"))

n.add_output("p9",  "t4",  Variable("x"))
n.add_output("p9",  "t5",  Variable("x"))

n.add_output("p12", "t8",  Variable("x"))
n.add_output("p12", "t10", Variable("x"))
n.add_output("p12", "t11", Variable("x"))

n.add_output("p18", "t9",  Variable("x"))

n.add_output("p17", "t12", Variable("x"))

n.add_output("p22", "t13", Variable("x"))
n.add_output("p23", "t14", Variable("x"))