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




# from snakes.nets import *

# n = PetriNet("peer_review")

# # Places (initial marking: p0=1, all others=0)
# n.add_place(Place("p0",  [0]))
# n.add_place(Place("p7",  []))
# n.add_place(Place("p8",  []))
# n.add_place(Place("p9",  []))
# n.add_place(Place("p12", []))
# n.add_place(Place("p16", []))
# n.add_place(Place("p17", []))
# n.add_place(Place("p18", []))
# n.add_place(Place("p19", []))
# n.add_place(Place("p20", []))
# n.add_place(Place("p22", []))
# n.add_place(Place("p23", []))

# # Transitions
# n.add_transition(Transition("t0"))
# n.add_transition(Transition("t2"))
# n.add_transition(Transition("t3"))
# n.add_transition(Transition("t4"))
# n.add_transition(Transition("t5"))
# n.add_transition(Transition("t6"))
# n.add_transition(Transition("t7"))
# n.add_transition(Transition("t8"))
# n.add_transition(Transition("t9"))
# n.add_transition(Transition("t10"))
# n.add_transition(Transition("t11"))
# n.add_transition(Transition("t12"))
# n.add_transition(Transition("t13"))
# n.add_transition(Transition("t14"))

# # Inputs (place -> transition)
# n.add_input("p0",  "t0",  Variable("x"))

# n.add_input("p16", "t6",  Variable("x"))
# n.add_input("p16", "t7",  Variable("x"))

# n.add_input("p7",  "t8",  Variable("x"))

# n.add_input("p19", "t2",  Variable("x"))
# n.add_input("p19", "t3",  Variable("x"))

# n.add_input("p8",  "t8",  Variable("x"))

# n.add_input("p20", "t4",  Variable("x"))
# n.add_input("p20", "t5",  Variable("x"))

# n.add_input("p9",  "t8",  Variable("x"))

# n.add_input("p12", "t9",  Variable("x"))

# n.add_input("p17", "t10", Variable("x"))
# n.add_input("p17", "t11", Variable("x"))

# n.add_input("p18", "t12", Variable("x"))
# n.add_input("p18", "t13", Variable("x"))
# n.add_input("p18", "t14", Variable("x"))

# # Outputs (transition -> place)
# n.add_output("p16", "t0",  Variable("x"))
# n.add_output("p19", "t0",  Variable("x"))
# n.add_output("p20", "t0",  Variable("x"))

# n.add_output("p7",  "t6",  Variable("x"))
# n.add_output("p7",  "t7",  Variable("x"))

# n.add_output("p8",  "t2",  Variable("x"))
# n.add_output("p8",  "t3",  Variable("x"))

# n.add_output("p9",  "t4",  Variable("x"))
# n.add_output("p9",  "t5",  Variable("x"))

# n.add_output("p12", "t8",  Variable("x"))
# n.add_output("p12", "t10", Variable("x"))
# n.add_output("p12", "t11", Variable("x"))

# n.add_output("p18", "t9",  Variable("x"))

# n.add_output("p17", "t12", Variable("x"))

# n.add_output("p22", "t13", Variable("x"))
# n.add_output("p23", "t14", Variable("x"))





from snakes.nets import *

n = PetriNet("N1_example_1")

# Places — removed p22, p17, p19, p20 to cut parallel branches
n.add_place(Place("p14", [0]))  # initial
n.add_place(Place("p1",  []))
n.add_place(Place("p2",  []))
n.add_place(Place("p10", []))
n.add_place(Place("p13", []))
n.add_place(Place("p12", []))
n.add_place(Place("p5",  []))
n.add_place(Place("p0",  []))
n.add_place(Place("p15", []))
n.add_place(Place("p16", []))
n.add_place(Place("p11", []))
n.add_place(Place("p3",  []))
n.add_place(Place("p18", []))
n.add_place(Place("p4",  []))
n.add_place(Place("p8",  []))
n.add_place(Place("p9",  []))
n.add_place(Place("p6",  []))
n.add_place(Place("p7",  []))  # final

# Transitions — kept the main spine, removed t31/t32 split, t35/t36 split
n.add_transition(Transition("t11"))  # init
n.add_transition(Transition("t21"))  # p1 -> p10
n.add_transition(Transition("t26"))  # p2 -> p3
n.add_transition(Transition("t31"))  # p10 -> p13+p21 (kept one branch)
n.add_transition(Transition("t41"))  # p13 -> p12
n.add_transition(Transition("t51"))  # p12 -> p5
n.add_transition(Transition("t61"))  # p5+p21 -> p15
n.add_transition(Transition("t71"))  # p15 -> p16
n.add_transition(Transition("t81"))  # p16 -> p11
n.add_transition(Transition("t36"))  # p3 -> p18
n.add_transition(Transition("t44"))  # p18 -> p4  (collapsed t44+t54)
n.add_transition(Transition("t66"))  # p4 -> p8
n.add_transition(Transition("t76"))  # p8 -> p9
n.add_transition(Transition("t91"))  # p11+p6 -> p7
n.add_transition(Transition("t82"))  # p9 -> p6

# Inputs
n.add_input("p14", "t11", Variable("x"))
n.add_input("p1",  "t21", Variable("x"))
n.add_input("p2",  "t26", Variable("x"))
n.add_input("p10", "t31", Variable("x"))
n.add_input("p13", "t41", Variable("x"))
n.add_input("p12", "t51", Variable("x"))
n.add_input("p5",  "t61", Variable("x"))
n.add_input("p15", "t71", Variable("x"))
n.add_input("p16", "t81", Variable("x"))
n.add_input("p3",  "t36", Variable("x"))
n.add_input("p18", "t44", Variable("x"))
n.add_input("p4",  "t66", Variable("x"))
n.add_input("p8",  "t76", Variable("x"))
n.add_input("p11", "t91", Variable("x"))
n.add_input("p6",  "t91", Variable("x"))
n.add_input("p9",  "t82", Variable("x"))

# Outputs
n.add_output("p1",  "t11", Variable("x"))
n.add_output("p2",  "t11", Variable("x"))
n.add_output("p10", "t21", Variable("x"))
n.add_output("p3",  "t26", Variable("x"))
n.add_output("p13", "t31", Variable("x"))
n.add_output("p12", "t41", Variable("x"))
n.add_output("p5",  "t51", Variable("x"))
n.add_output("p15", "t61", Variable("x"))
n.add_output("p16", "t71", Variable("x"))
n.add_output("p11", "t81", Variable("x"))
n.add_output("p18", "t36", Variable("x"))
n.add_output("p4",  "t44", Variable("x"))
n.add_output("p8",  "t66", Variable("x"))
n.add_output("p9",  "t76", Variable("x"))
n.add_output("p6",  "t82", Variable("x"))
n.add_output("p7",  "t91", Variable("x"))