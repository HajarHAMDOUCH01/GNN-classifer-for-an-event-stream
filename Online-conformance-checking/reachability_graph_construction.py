from snakes.nets import *

n = PetriNet("experimentation_example")

# Places
n.add_place(Place("p0", [0]))
n.add_place(Place("p1", []))
n.add_place(Place("p2", []))
n.add_place(Place("p3", []))
n.add_place(Place("p4", []))
n.add_place(Place("p5", []))
n.add_place(Place("p6", []))

# Transitions (sans Expression pour les noms simples)
n.add_transition(Transition("t1"))
n.add_transition(Transition("t2"))
n.add_transition(Transition("t3"))
n.add_transition(Transition("t4"))
n.add_transition(Transition("t5"))
n.add_transition(Transition("t6"))
n.add_transition(Transition("t7"))
n.add_transition(Transition("t8"))

# Inputs
n.add_input("p0", "t1", Variable("x"))
n.add_input("p1", "t2", Variable("x"))
n.add_input("p2", "t3", Variable("x"))
n.add_input("p4", "t4", Variable("x"))
n.add_input("p1", "t4", Variable("x"))
n.add_input("p3", "t5", Variable("x"))
n.add_input("p4", "t5", Variable("x"))
n.add_input("p5", "t6", Variable("x"))
n.add_input("p5", "t7", Variable("x"))
n.add_input("p5", "t8", Variable("x"))

# Outputs
n.add_output("p1", "t1", Variable("x"))
n.add_output("p2", "t1", Variable("x"))
n.add_output("p3", "t2", Variable("x"))
n.add_output("p4", "t3", Variable("x"))
n.add_output("p5", "t4", Variable("x"))
n.add_output("p5", "t5", Variable("x"))
n.add_output("p1", "t6", Variable("x"))
n.add_output("p2", "t6", Variable("x"))
n.add_output("p6", "t7", Variable("x"))
n.add_output("p6", "t8", Variable("x"))

# modes = n.transition("t1").modes()
# print(f"modes of t1: {modes}")
# if modes:
#     n.transition("t1").fire(modes[0])

# print(f"marking after firing t1: {n.get_marking()}")
# output : 
# modes of t1: [Substitution(x=0)]
# marking after firing t1: {p1={0}, p2={0}}


# un exemple de process conforme possible
# tous les process possibles composent reachability graph

# initial_marking = n.get_marking()
# print("initial marking :", initial_marking)


# t = n.transition("t1")
# modes = t.modes()
# t.fire(modes.pop())
# print(f"marking after firing t1: {n.get_marking()}")

# t = n.transition("t2")
# modes = t.modes()
# t.fire(modes.pop())
# print(f"marking after firing t2: {n.get_marking()}")

# t = n.transition("t3")
# modes = t.modes()
# t.fire(modes.pop())
# print(f"marking after firing t3: {n.get_marking()}")

# t = n.transition("t5")
# modes = t.modes()
# t.fire(modes.pop())
# print(f"marking after firing t5: {n.get_marking()}")


# t = n.transition("t7")
# modes = t.modes()
# t.fire(modes.pop())
# print(f"marking after firing t7: {n.get_marking()}")

from collections import deque

def build_reachability_graph(net):
    graph = {}  # {marking: [(transition, mode, next_marking)]}
    
    initial = net.get_marking()
    queue = deque([initial])
    visited = {initial}
    
    while queue:
        marking = queue.popleft()
        graph[marking] = []
        
        net.set_marking(marking)  # restaure le marquage courant
        
        for t_name in [t.name for t in net.transition()]:
            t = net.transition(t_name)
            net.set_marking(marking)
            for mode in t.modes():
                print("Firing transition:", t_name, "with mode:", mode)
                # tire la transition
                net.set_marking(marking)
                t.fire(mode)
                next_marking = net.get_marking()
                
                graph[marking].append((t_name, mode, next_marking))
                
                if next_marking not in visited:
                    visited.add(next_marking)
                    queue.append(next_marking)
    
    return graph


reachability_graph = build_reachability_graph(n)
print("\n=== Reachability Graph ===")
for marking, transitions in reachability_graph.items():
    print(f"\nFrom: {marking}")
    for t_name, mode, next_marking in transitions:
        print(f"  --[{t_name}]--> {next_marking}")

print(f"\nTotal states: {len(reachability_graph)}")