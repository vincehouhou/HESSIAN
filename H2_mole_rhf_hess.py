
# ************************************************************************
# File:    H2_mole_rhf_hess.py
# Purpose: Hessian Analysis on the ground-state of H2 molecule in restricted Hartree-Fock model, including energy correction step
# Version: FEniCS 1.4.0
# Author:  Houdong Hu
# ************************************************************************


from dolfin import *
from math import *
import numpy as np
import scipy as sc
import scipy.linalg
import datetime
import sys

# calculate the determinant of Jacobian 

def getJ(coor1,coor2,coor3,coor4):
     J=np.array([[coor2[0]-coor1[0],coor3[0]-coor1[0],coor4[0]-coor1[0]],[coor2[1]-coor1[1],coor3[1]-coor1[1],coor4[1]-coor1[1]],[coor2[2]-coor1[2],coor3[2]-coor1[2],coor4[2]-coor1[2]]])
     return abs(np.linalg.det(J))

# 1st order Gauss Quadrature

def getQ1(a,coor1,coor2,coor3,coor4,det):
     verc=(coor1+coor2+coor3+coor4)/4.0
     addi=np.zeros((4,4))
     addi.fill(a(verc)*det/16.0/6.0)
     return addi

# 2nd order Gauss Quadrature

def getQ2(a,coor1,coor2,coor3,coor4,det):
     ca=0.585410196624969
     cb=0.138196601125011
     verc1=ca*coor1+cb*coor2+cb*coor3+cb*coor4
     verc2=cb*coor1+ca*coor2+cb*coor3+cb*coor4
     verc3=cb*coor1+cb*coor2+ca*coor3+cb*coor4
     verc4=cb*coor1+cb*coor2+cb*coor3+ca*coor4
     addi=np.zeros((4,4))
     addi[0,0]=(ca*ca*a(verc1)+cb*cb*a(verc2)+cb*cb*a(verc3)+cb*cb*a(verc4))*det/24.0
     addi[1,1]=(cb*cb*a(verc1)+ca*ca*a(verc2)+cb*cb*a(verc3)+cb*cb*a(verc4))*det/24.0
     addi[2,2]=(cb*cb*a(verc1)+cb*cb*a(verc2)+ca*ca*a(verc3)+cb*cb*a(verc4))*det/24.0
     addi[3,3]=(cb*cb*a(verc1)+cb*cb*a(verc2)+cb*cb*a(verc3)+ca*ca*a(verc4))*det/24.0
     addi[0,1]=(ca*cb*a(verc1)+ca*cb*a(verc2)+cb*cb*a(verc3)+cb*cb*a(verc4))*det/24.0
     addi[0,2]=(ca*cb*a(verc1)+cb*cb*a(verc2)+ca*cb*a(verc3)+cb*cb*a(verc4))*det/24.0
     addi[0,3]=(ca*cb*a(verc1)+cb*cb*a(verc2)+cb*cb*a(verc3)+ca*cb*a(verc4))*det/24.0
     addi[1,2]=(cb*cb*a(verc1)+ca*cb*a(verc2)+ca*cb*a(verc3)+cb*cb*a(verc4))*det/24.0
     addi[1,3]=(cb*cb*a(verc1)+ca*cb*a(verc2)+cb*cb*a(verc3)+ca*cb*a(verc4))*det/24.0
     addi[2,3]=(cb*cb*a(verc1)+cb*cb*a(verc2)+ca*cb*a(verc3)+ca*cb*a(verc4))*det/24.0
     addi[1,0]=addi[0,1]
     addi[2,0]=addi[0,2]
     addi[3,0]=addi[0,3]
     addi[2,1]=addi[1,2]
     addi[3,1]=addi[1,3]
     addi[3,2]=addi[2,3]
     return addi

# get 1/R constant between each pair of elements

def get1oR(coori1,coori2,coori3,coori4,coorj1,coorj2,coorj3,coorj4,delta):
     verci=(coori1+coori2+coori3+coori4)/4.0
     vercj=(coorj1+coorj2+coorj3+coorj4)/4.0
     return 1.0/(((verci[0]-vercj[0])**2.0+(verci[1]-vercj[1])**2.0+(verci[2]-vercj[2])**2.0)**0.5+delta)


bl=0.7

mesh = BoxMesh(-50,-50,-50,50,50,50,2,2,2)
origin1 = Point(-bl,.0,.0)
origin2 = Point(bl,.0,.0)

# construct priori adaptive mesh

for j in range(20):
    cell_markers = CellFunction("bool",mesh)
    cell_markers.set_all(False)
    min1 = 1000
    min2 = 1000
    for cell in cells(mesh):
        p = cell.midpoint()
        r =((p[0]-origin1[0])**2+(p[1]-origin1[1])**2+(p[2]-origin1[2])**2)**0.5
        if r<min1:
           min1=r
        r =((p[0]-origin2[0])**2+(p[1]-origin2[1])**2+(p[2]-origin2[2])**2)**0.5
        if r<min2:
           min2=r
    
    for cell in cells(mesh):
        p = cell.midpoint()
        r =((p[0]-origin1[0])**2+(p[1]-origin1[1])**2+(p[2]-origin1[2])**2)**0.5
        if r<=min1:
            cell_markers[cell]=True
        r =((p[0]-origin2[0])**2+(p[1]-origin2[1])**2+(p[2]-origin2[2])**2)**0.5
        if r<=min2:
            cell_markers[cell]=True
   
    mesh = refine(mesh, cell_markers)


for k in range(0):
    cell_markers = CellFunction("bool",mesh)
    cell_markers.set_all(True)
    mesh = refine(mesh, cell_markers)

cmin = 100
for cell in cells(mesh):
    p = cell.midpoint()
    r =((p[0]-origin1[0])**2+(p[1]-origin1[1])**2+(p[2]-origin1[2])**2)**0.5
    if r<cmin:
       cmin=r
print cmin
print mesh.coordinates().shape
print mesh.coordinates().shape

V = FunctionSpace(mesh, "Lagrange", 1)
Vc = FunctionSpace(mesh, "Lagrange", 1)
v2d_map = vertex_to_dof_map(V)
d2v_map=dof_to_vertex_map(V)

delta=cmin
v_ext_corr=Expression("1.0/(pow(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5))+1.0/(pow(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5))",bl=bl)#+delta)",delta=delta)
v_ext1=Expression("1.0/pow(pow(x[0]-0.0,2)+pow(x[1]-0.0, 2)+pow(x[2]-0.0,2),0.5)")

def boundary(x, on_boundary):
    return on_boundary

# Define boundary condition

u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)
bc1 = DirichletBC(V, v_ext1, boundary)


# Define test function and initial condition (STO-3G)

u = TrialFunction(V)
v = TestFunction(V)

alpha11=3.43525091
alpha12=0.62391373
alpha13=0.16885540
tmp11=Expression("(pow(2*alpha11/pi,3.0/4.0))*exp(-alpha11*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha11=alpha11,bl=bl)
tmp12=Expression("(pow(2*alpha12/pi,3.0/4.0))*exp(-alpha12*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha12=alpha12,bl=bl)
tmp13=Expression("(pow(2*alpha13/pi,3.0/4.0))*exp(-alpha13*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha13=alpha13,bl=bl)
rx1s=0.15432897*(interpolate(tmp11,V)).vector().array()+0.53532814*(interpolate(tmp12,V)).vector().array()+0.44463454*(interpolate(tmp13,V)).vector().array()

tmp11=Expression("(pow(2*alpha11/pi,3.0/4.0))*exp(-alpha11*(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha11=alpha11,bl=bl)
tmp12=Expression("(pow(2*alpha12/pi,3.0/4.0))*exp(-alpha12*(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha12=alpha12,bl=bl)
tmp13=Expression("(pow(2*alpha13/pi,3.0/4.0))*exp(-alpha13*(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha13=alpha13,bl=bl)
rx2s=0.15432897*(interpolate(tmp11,V)).vector().array()+0.53532814*(interpolate(tmp12,V)).vector().array()+0.44463454*(interpolate(tmp13,V)).vector().array()


r1s=-0.5
r1s_p=0.0
r2s=-0.5
r2s_p=0.0
energy=-2.5
energy_p=0.0

tmp1=Function(V)
tmp1.vector()[:]=rx1s+rx2s
frx1s=Function(V)
v1s = Function(V)
threshold=1e-7
tmpv=tmp1.vector().array()
   
while abs(energy-energy_p)>threshold:

    time0=datetime.datetime.now()

    # normalization

    nor1=assemble(inner(tmp1,tmp1)*dx)**0.5
    frx1s.vector()[:]=tmp1.vector().array()/nor1

    # coulomb potential 

    b1s=frx1s*frx1s*v*dx
    a=0.25/pi*inner(nabla_grad(u), nabla_grad(v))*dx

    problem = LinearVariationalProblem(a, b1s, v1s, bcs=bc1)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "cg"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()
    
    # iterate Helmholtz equation

    hela1s=0.5*inner(nabla_grad(u), nabla_grad(v))*dx-u*r1s*v*dx
    helb1s=(-v1s*frx1s+v_ext_corr*frx1s)*v*dx

    problem = LinearVariationalProblem(hela1s, helb1s, tmp1, bcs=bc)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "cg"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()

    # energy correction

    t1=assemble(inner(-v1s*frx1s+v_ext_corr*frx1s,frx1s-tmp1)*dx)
    n1=assemble(tmp1*tmp1*dx)
    r1s_p=r1s
    r1s=r1s+t1/n1
 
    energy_T=assemble(inner(grad(frx1s),grad(frx1s))*dx)
    energy_H=assemble(v1s*frx1s*frx1s*dx)
    energy_ex=-2.0*assemble(v_ext_corr*frx1s*frx1s*dx)
    energy_nu=1.0/(2.0*bl)
    energy_p=energy
    energy=energy_T+energy_H+energy_ex+energy_nu
    
    time1=datetime.datetime.now()

    print r1s,r2s,energy_p,energy,energy-energy_p,(time1-time0).total_seconds() 



a1=Expression('1.0')

parameters.linear_algebra_backend="uBLAS"
a=0.5*inner(nabla_grad(u),nabla_grad(v))*dx-u*v_ext_corr*v*dx+u*v1s*v*dx-u*r1s*v*dx
m=u*v*dx

A=assemble(a)
M=assemble(m)

rows,cols,values=A.data()
rowsm,colsm,valuesm=M.data()


# relationship between vertex and element                        

e_v=mesh.cells()
noe=e_v.shape[0]

# coordinates of vertex                                          

coor = mesh.coordinates()
dof=coor.shape[0]


# transform from sparse matrix to dense matrix                   

A1=np.zeros((dof,dof))
M1=np.zeros((dof,dof))
for i in range(dof):
    ii=d2v_map[i]
    for j in range(rows[i],rows[i+1]):
        jj=d2v_map[cols[j]]  
        A1[ii,jj]=values[j]
for i in range(dof):
    ii=d2v_map[i]
    for j in range(rowsm[i],rowsm[i+1]):
        jj=d2v_map[cols[j]]
        M1[ii,jj]=valuesm[j]


# normalization

nor1=assemble(inner(tmp1,tmp1)*dx)**0.5
frx1s.vector()[:]=tmp1.vector().array()/nor1
v1=frx1s.vector().array()
tmpv=tmp1.vector().array()


# element stiffness matrix
ca=0.585410196624969
cb=0.138196601125011
add=np.zeros((4,4))
add[0,0]=(ca*ca+3.0*cb*cb)/24.0
add[1,1]=(ca*ca+3.0*cb*cb)/24.0
add[2,2]=(ca*ca+3.0*cb*cb)/24.0
add[3,3]=(ca*ca+3.0*cb*cb)/24.0
add[0,1]=(ca*cb+ca*cb+cb*cb+cb*cb)/24.0
add[0,2]=(ca*cb+cb*cb+ca*cb+cb*cb)/24.0
add[0,3]=(ca*cb+cb*cb+cb*cb+ca*cb)/24.0
add[1,2]=(cb*cb+ca*cb+ca*cb+cb*cb)/24.0
add[1,3]=(cb*cb+ca*cb+cb*cb+ca*cb)/24.0
add[2,3]=(cb*cb+cb*cb+ca*cb+ca*cb)/24.0
add[1,0]=add[0,1]
add[2,0]=add[0,2]
add[3,0]=add[0,3]
add[2,1]=add[1,2]
add[3,1]=add[1,3]
add[3,2]=add[2,3]

cmax=0
time1=datetime.datetime.now()

# assembly stiffness matrix for exact exchange operator

for i in range(noe):
    time0=time1
    time1=datetime.datetime.now()
    sys.stdout.flush()
    deti=getJ(coor[e_v[i,0]],coor[e_v[i,1]],coor[e_v[i,2]],coor[e_v[i,3]])
    addi=add*deti
    for j in range(noe):
          detj=getJ(coor[e_v[j,0]],coor[e_v[j,1]],coor[e_v[j,2]],coor[e_v[j,3]])      
          addj=add*detj
          oneoR=get1oR(coor[e_v[i,0]],coor[e_v[i,1]],coor[e_v[i,2]],coor[e_v[i,3]],coor[e_v[j,0]],coor[e_v[j,1]],coor[e_v[j,2]],coor[e_v[j,3]],delta)
          for m in range(4):
               for n in range(4):
                    for m1 in range(4):
                         for n1 in range(4):
                             tmp=2.0*oneoR*v1[v2d_map[e_v[i,m1]]]*v1[v2d_map[e_v[j,n1]]]*addi[m,m1]*addj[n,n1]
                             A1[e_v[i,m],e_v[j,n]]+=tmp
                             cmax=max(cmax,tmp)   
    print i,(time1-time0).total_seconds(),deti 
   

# calculate the eigenvalue spectrum of Hessian matrix

print cmax
w,v=sc.linalg.eig(a=A1,b=M1)
print w
w1,v1=sc.linalg.eigh(a=A1,b=M1)
print w1

# output the negative eigenvalues, and their corresponding directions

for i in range(w.size):
    if w[i]<0:
        test1=np.ones((dof,))
        for j in range(dof):
            test1[v2d_map[j]]=v[j,i]
        testf1=Function(V)
        testf1.vector()[:]=test1
        nor=assemble(inner(testf1,testf1)*dx)**0.5
        testf1.vector()[:]=test1/nor
        print w[i],assemble(inner(frx1s,testf1)*dx),assemble(inner(frx1s,frx1s)*dx),assemble(inner(testf1,testf1)*dx)
