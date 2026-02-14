
# TO DOOOOO : think how to estimate a marking for a non-conform transition (wether coming from a conformant state or non-conformant state)
"""
marking estimation <=> 'very not likely jump' => very high cost : 
distance between the current amrking and marking before a non-conformant transition
"""


# movements required to align to a conformant sequence should be telling the relashionship between paths
# optimization task : a model shpuld learn the movements and relashionships between movements  

#...............................................................................
# Loss = Erreur de Position + coefficient * Nombre de saut des marking conformes
#...............................................................................

#...............................................................................
# Loss de position = 1 - cos(position estimée, position cible)
#...............................................................................


