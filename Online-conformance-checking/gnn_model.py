
# TO DOOOOO : think how to estimate a marking for a non-conform transition (wether coming from a conformant state or non-conformant state)
# do something that will let you estimate the least effort required to go back to a conformant alignement and consider the cost and then continue the transitions next
# only then you can classify


"""
marking estimation <=> 'very not likely jump' => very high cost : 
distance between the current amrking and marking before a non-conformant transition
"""

#...............................................................................
# Loss = Erreur de Position + coefficient * Nombre de saut des marking conformes
#...............................................................................

#...............................................................................
# Loss de position = 1 - cos(position estimée, position cible)
#...............................................................................


