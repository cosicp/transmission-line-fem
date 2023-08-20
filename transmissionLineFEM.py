# Transmission line FEM example
# Author: Petar Cosic <cpetar112@gmail.com>
# Date: August 2023
# Python 3.11.3
# -------------------------------------------

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from node_data import *

# Cross-section characteristics
Ds = 20  # [mm] Outer diameter of the pipe
t = 2  # [mm] Pipe wall thickness
# Material characteristics
E = 200e9  # [Pa] Modulus of elasticity

# Boundary conditions
# bc(node, 1:3) = [Tx, Ty, Tz] 
# 1 - free
# 0 - constrained
nn = np.size(node_coord,0)  # number of nodes
bc = np.zeros([nn,3])
bc[0:4] = [1,1,1]

# Force vector
F = np.zeros([nn,3])
F[29:33] = [0,0,-20000]

# Options
displacement_magnification = 10

# Initialization of vectors and matrices
ne = np.size(el_node_num, 0)  # number of elements
dof = 3*nn  # number degrees of freedom
D = np.zeros([dof, 1])  # displacement vector
F = np.reshape(F,[1,np.size(F,0)*np.size(F,1)])[0]  # force vector
K = np.zeros([dof, dof])  # stiffness matrix 
sigma = np.zeros([ne, 1]) # stress vector

# Degrees of freedom of the nodes
bc_reshaped = np.reshape(bc,[1,np.size(bc)])[0]
rdof = [i for i, x in enumerate(bc_reshaped) if x==1]
fdof = [i for i, x in enumerate(bc_reshaped) if x==0]

# Cross-sectional characteristics (round pipe)
Ds = Ds/1000  # [m] 
t = t/1000    # [m]
A = (Ds**2-(Ds-2*t)**2)*np.pi/4  # [m^2] Cross-sectional area

# Node coordinates in [m]
node_coord_m = np.divide(node_coord,1000)  

# Forming the global stiffness matrix
for i in range(0,ne):
  j = el_node_num[i]
  edof = np.subtract([3*j[0]-2, 3*j[0]-1, 3*j[0], # Degrees of freedom of an element
                      3*j[1]-2, 3*j[1]-1, 3*j[1]],1)           
  j = np.subtract(j,1)
  # Length of the element
  L_e = np.sqrt((node_coord_m[j[1]][0]-node_coord_m[j[0]][0])**2+(node_coord_m[j[1]][1]-node_coord_m[j[0]][1])**2+(node_coord_m[j[1]][2]-node_coord_m[j[0]][2])**2) 
  # Determining cosines of angles
  CXx = (node_coord_m[j[1]][0]-node_coord_m[j[0]][0])/L_e   # cos(teta)
  CYx = (node_coord_m[j[1]][1]-node_coord_m[j[0]][1])/L_e   # cos(teta)
  CZx = (node_coord_m[j[1]][2]-node_coord_m[j[0]][2])/L_e   # cos(teta)
  # The stiffness matrix of the elements in the local coordinate system of the elements
  k_e = (E*A)/L_e
  # Segment of the transformation matrix
  r = [[CXx*CXx, CXx*CYx, CXx*CZx],
       [CYx*CXx, CYx*CYx, CYx*CZx],
       [CZx*CXx, CZx*CYx, CZx*CZx]]
  # Forming the full transformation matrix
  T = np.concatenate((np.concatenate((r, np.multiply(r,-1)),1) , np.concatenate((np.multiply(r,-1),r),1)),0)
  # The stiffness matrix of the elements in the global coordinate system
  [ind1,ind2] = np.meshgrid(edof,edof)
  K[ind2,ind1] = np.add(K[ind2,ind1], k_e*T)

# Solving the system
[ind1,ind2] = np.meshgrid(fdof,fdof)
D1 = np.matmul(np.linalg.inv(K[ind2,ind1]),F[fdof])
D = np.transpose(D)[0]
D[fdof] = D1 # Full displacement vector
D = np.transpose([D])
Dn = np.reshape(D,[int(np.size(D,0)/3),3])
F1 = np.matmul(K,D) # Force vector
R = np.transpose([np.transpose(F1)[0][rdof]]) # Reaction vector

# Determining forces in rods
for i in range(0,ne):
  j = el_node_num[i]
  edof = np.subtract([3*j[0]-2, 3*j[0]-1, 3*j[0], # degrees of freedom of an element
                      3*j[1]-2, 3*j[1]-1, 3*j[1]],1)    
  j = np.subtract(j,1)
  # Element length
  L_e = np.sqrt((node_coord_m[j[1]][0]-node_coord_m[j[0]][0])**2+(node_coord_m[j[1]][1]-node_coord_m[j[0]][1])**2+(node_coord_m[j[1]][2]-node_coord_m[j[0]][2])**2)
  # Determining cosines of angles
  CXx = (node_coord_m[j[1]][0]-node_coord_m[j[0]][0])/L_e  # cos(teta)
  CYx = (node_coord_m[j[1]][1]-node_coord_m[j[0]][1])/L_e  # cos(teta)
  CZx = (node_coord_m[j[1]][2]-node_coord_m[j[0]][2])/L_e  # cos(teta)
  sigma[i][0] = np.dot(np.multiply(E/L_e,[-CXx, -CYx, -CZx, CXx, CYx, CZx]),D[edof]) # [Pa] Normal stress

# Forces in rods
Q = sigma*A  # [N] Rod forces

# Displaying results in console
print('        Displacements         ')
print('------------------------------')
print('  node  |  comp.  |  displacement [mm]');
counter = 1
for i in range(0,len(D),3):
    print(f'   {counter}    |    1    |    {round(D[i][0]*1000,5)}')
    print(f'   {counter}    |    2    |    {round(D[i+1][0]*1000,5)}')
    print(f'   {counter}    |    3    |    {round(D[i+2][0]*1000,5)}')
    counter+=1
    
print('\n')
counter = 1

print('        Reactions         ')
print('--------------------------')
print('  node  |  comp.  |  Reaction [N]');
for i in range(0,len(R),3):
   print(f'   {counter}    |    1    |    {round(R[i][0],5)}')
   print(f'   {counter}    |    2    |    {round(R[i+1][0],5)}')
   print(f'   {counter}    |    3    |    {round(R[i+2][0],5)}')
   counter+=1
   
print('\n')
print('      Forces in rods      ')
print('--------------------------')
print('   rod  |   Force [N]');
for i in range(0,len(Q)):
   print(f'   {el_node_num[i][0]}-{el_node_num[i][1]}  |  {round(Q[i][0],5)}')

# Ploting 
node_coords_deformed = node_coord+Dn*1000*displacement_magnification # Deformed model node coordinates

rds = np.transpose(np.vstack([np.vstack([node_coord[np.subtract(el_node_num[:, 0],1),0],node_coord[np.subtract(el_node_num[:, 1],1),0]]),
                              np.vstack([node_coord[np.subtract(el_node_num[:, 0],1),1],node_coord[np.subtract(el_node_num[:, 1],1),1]]),
                              np.vstack([node_coord[np.subtract(el_node_num[:, 0],1),2],node_coord[np.subtract(el_node_num[:, 1],1),2]])]))

rdsd = np.transpose(np.vstack([np.vstack([node_coords_deformed[np.subtract(el_node_num[:, 0],1),0],node_coords_deformed[np.subtract(el_node_num[:, 1],1),0]]),
                              np.vstack([node_coords_deformed[np.subtract(el_node_num[:, 0],1),1],node_coords_deformed[np.subtract(el_node_num[:, 1],1),1]]),
                              np.vstack([node_coords_deformed[np.subtract(el_node_num[:, 0],1),2],node_coords_deformed[np.subtract(el_node_num[:, 1],1),2]])]))

fig,(ax1,ax2) =  plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
fig.figsize = (2,5)
fig.dpi = 200

for i in range(len(rds)):
  ax1.plot3D(rds[i,0:2],rds[i,2:4],rds[i,4:6],color = '#0022ff',linewidth = 0.9); # undeformed model
  ax1.scatter(rds[i,0:2],rds[i,2:4],rds[i,4:6],color = '#ff2200',s = 3); # undeformed model nodes
  ax2.plot3D(rds[i,0:2],rds[i,2:4],rds[i,4:6],color = '#ff0000',linewidth = 0.9); # undeformed model
  ax2.plot3D(rdsd[i,0:2],rdsd[i,2:4],rdsd[i,4:6],color = '#00ff00',linestyle = '--',linewidth = 0.8) # deformed model

ax1.quiver(node_coord[29:33,0], node_coord[29:33,1], node_coord[29:33,2] ,0, 0, -700,color = 'black')
ax1.set_title("Original model")
ax2.set_title("Deformed model")
ax1.set_aspect('equal', 'box')
ax2.set_aspect('equal', 'box')
ax1.view_init(27, 38, 0)
ax2.view_init(27, 38, 0)
ax1.axis('off')
ax2.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0.2, hspace=0)

# Toggle show node numbers button
check = CheckButtons(fig.add_axes([0.7, 0.05, 0.1, 0.05]), ['nodes'])
class toggleNodesClass:
  def __init__(self):
    self.toggle_flag = True
    self.text = []
    
  def toggleNodes(self, event):
    if self.toggle_flag:
      for i in range(len(node_coord)): 
        self.text.append(ax1.text(node_coord[i,0]+50, node_coord[i,1]+50,node_coord[i,2]+50\
      , f"{i+1}",size = 6,verticalalignment='bottom',horizontalalignment = 'left'))
      plt.draw()
      self.toggle_flag = False
    else:
      self.toggle_flag = True
      [text.remove() for text in self.text]
      self.text = []
      plt.draw()

callback_button = toggleNodesClass()
check.on_clicked(callback_button.toggleNodes)

# Uncomment for circural animation
#for i in range(360-38):
#    ax1.view_init(27, 38+i, 0)
#    ax2.view_init(27, 38+i, 0)
#    plt.draw()
#    plt.pause(.01)

plt.show()
