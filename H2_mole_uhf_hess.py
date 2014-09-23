
# ************************************************************************
# File:    H2_mole_uhf_hess.py
# Purpose: Hessian Analysis on the ground-state of H2 molecule in unrestricted Hartree-Fock model, including energy correction step
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

# get 1/R constant between a pair of finite elements

def get1oR(coori1,coori2,coori3,coori4,coorj1,coorj2,coorj3,coorj4,delta):
    verci=(coori1+coori2+coori3+coori4)/4.0
    vercj=(coorj1+coorj2+coorj3+coorj4)/4.0
    return 1.0/(((verci[0]-vercj[0])**2.0+(verci[1]-vercj[1])**2.0+(verci[2]-vercj[2])**2.0)**0.5+delta)



bl=2.8

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

for k in range(1):
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

V = FunctionSpace(mesh, "Lagrange", 1)
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
tmp1.vector()[:]=(rx1s+rx2s)/2.0
frx1s=Function(V)
v1s = Function(V)
tmp2=Function(V)
tmp2.vector()[:]=(rx1s+rx2s)/2.0
frx2s=Function(V)
v2s = Function(V)
threshold=1e-7


while abs(energy-energy_p)>threshold:

    time0=datetime.datetime.now()

    # normalization

    nor1=assemble(inner(tmp1,tmp1)*dx)**0.5
    frx1s.vector()[:]=tmp1.vector().array()/nor1
    nor2=assemble(inner(tmp2,tmp2)*dx)**0.5
    frx2s.vector()[:]=tmp2.vector().array()/nor2

    # coulomb potential 

    b1s=frx1s*frx1s*v*dx
    a=0.25/pi*inner(nabla_grad(u), nabla_grad(v))*dx

    problem = LinearVariationalProblem(a, b1s, v1s, bc1)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()
    
    b2s=frx2s*frx2s*v*dx

    problem = LinearVariationalProblem(a, b2s, v2s, bc1)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()
 
    # iterate Helmholtz equation

    hela1s=0.5*inner(nabla_grad(u), nabla_grad(v))*dx-u*r1s*v*dx
    helb1s=(-v2s*frx1s+v_ext_corr*frx1s)*v*dx
    hela2s=0.5*inner(nabla_grad(u), nabla_grad(v))*dx-u*r2s*v*dx
    helb2s=(-v1s*frx2s+v_ext_corr*frx2s)*v*dx

    problem = LinearVariationalProblem(hela1s, helb1s, tmp1, bc)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()
    problem = LinearVariationalProblem(hela2s, helb2s, tmp2, bc)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()
   
    # energy correction

    t1=assemble(inner(-v2s*frx1s+v_ext_corr*frx1s,frx1s-tmp1)*dx)
    n1=assemble(tmp1*tmp1*dx)
    r1s_p=r1s
    r1s=r1s+t1/n1

    t2=assemble(inner(-v1s*frx2s+v_ext_corr*frx2s,frx2s-tmp2)*dx)
    n2=assemble(tmp2*tmp2*dx)
    r2s_p=r2s
    r2s=r2s+t2/n2
 
    energy_T=0.5*assemble(inner(grad(frx1s),grad(frx1s))*dx)+0.5*assemble(inner(grad(frx2s),grad(frx2s))*dx)
    energy_H=0.5*assemble(v2s*frx1s*frx1s*dx)+0.5*assemble(v1s*frx2s*frx2s*dx)
    energy_ex=-assemble(v_ext_corr*frx1s*frx1s*dx)-assemble(v_ext_corr*frx2s*frx2s*dx)
    energy_nuc=1.0/(2.0*bl)
    energy_p=energy
    energy=energy_T+energy_H+energy_ex+energy_nuc
    
    time1=datetime.datetime.now()

    # output in each step

    print r1s,r2s,energy_p,energy,energy-energy_p,(time1-time0).total_seconds(),assemble(frx1s*frx1s*dx),assemble(frx2s*frx2s*dx),assemble(frx1s*frx2s*dx)
  
    
parameters.linear_algebra_backend="uBLAS"
a1=0.5*inner(nabla_grad(u),nabla_grad(v))*dx-u*v_ext_corr*v*dx+u*v2s*v*dx-u*r1s*v*dx
a2=0.5*inner(nabla_grad(u),nabla_grad(v))*dx-u*v_ext_corr*v*dx+u*v1s*v*dx-u*r2s*v*dx
m=u*v*dx

A1=assemble(a1)
A2=assemble(a2)
M=assemble(m)

rows1,cols1,values1=A1.data()
rows2,cols2,values2=A2.data()
rowsm,colsm,valuesm=M.data()


# relationship between vertex and element                        

e_v=mesh.cells()
noe=e_v.shape[0]

# coordinates of vertex                                          

coor = mesh.coordinates()
dof=coor.shape[0]



# transform from sparse matrix to dense matrix                   

A11=np.zeros((dof,dof))
M11=np.zeros((dof,dof))
A22=np.zeros((dof,dof))
A12=np.zeros((dof,dof))
for i in range(dof):
    ii=d2v_map[i]
    for j in range(rows1[i],rows1[i+1]):
        jj=d2v_map[cols1[j]]
        A11[ii,jj]=values1[j]
for i in range(dof):
    ii=d2v_map[i]
    for j in range(rows2[i],rows2[i+1]):
        jj=d2v_map[cols1[j]]
        A22[ii,jj]=values2[j]
for i in range(dof):
    ii=d2v_map[i]
    for j in range(rowsm[i],rowsm[i+1]):
        jj=d2v_map[cols1[j]]
        M11[ii,jj]=valuesm[j]

# normalization

nor1=assemble(inner(tmp1,tmp1)*dx)**0.5
frx1s.vector()[:]=tmp1.vector().array()/nor1
v1=frx1s.vector().array()
nor2=assemble(inner(tmp2,tmp2)*dx)**0.5
frx2s.vector()[:]=tmp2.vector().array()/nor2
v2=frx2s.vector().array()

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

print max(v1),max(v2)
time1=datetime.datetime.now()

# assembly stiffness matrix of exact exchange operator

for i in range(noe):
    time0=time1
    time1=datetime.datetime.now()
    sys.stdout.flush()
    deti=getJ(coor[e_v[i,0]],coor[e_v[i,1]],coor[e_v[i,2]],coor[e_v[i,3]])
    addi=add*deti
    print i,(time1-time0).total_seconds() 
    for j in range(noe):
        detj=getJ(coor[e_v[j,0]],coor[e_v[j,1]],coor[e_v[j,2]],coor[e_v[j,3]])
        addj=add*detj
        oneoR=get1oR(coor[e_v[i,0]],coor[e_v[i,1]],coor[e_v[i,2]],coor[e_v[i,3]],coor[e_v[j,0]],coor[e_v[j,1]],coor[e_v[j,2]],coor[e_v[j,3]],delta)
        for m in range(4):
            for n in range(4):
                for m1 in range(4):
                    for n1 in range(4):
                        tmp=(2.0*oneoR*v1[v2d_map[e_v[i,m1]]]*v2[v2d_map[e_v[j,n1]]]*addi[m,m1]*addj[n,n1])
                        A12[e_v[i,m],e_v[j,n]]+=tmp



# Hessian matrix and mass matrix setup

H=np.zeros((2*dof,2*dof))
M=np.zeros((2*dof,2*dof))

for i in range(dof):
    for j in range(dof):
        H[i,j]=A11[i,j]
        H[dof+i,dof+j]=A22[i,j]
        H[i,dof+j]=A12[i,j]
        H[dof+j,i]=A12[i,j]
        M[i,j]=M11[i,j]
        M[dof+i,dof+j]=M11[i,j]


# calculate eigenvalue spectrum of Hessian matrix

w,v=sc.linalg.eig(a=H,b=M)
print w
w1,v1=sc.linalg.eigh(a=H,b=M)
print w1


# output the negative eigenvalues of Hessian matrix, and analyze its eigenvector direction

for i in range(w.size):
    if w[i]<0:
        test1=np.ones((dof,))
        test2=np.ones((dof,))
        for j in range(dof):
            test1[v2d_map[j]]=v[j,i]
            test2[v2d_map[j]]=v[dof+j,i]
        testf1=Function(V)
        testf1.vector()[:]=test1
        nor=assemble(inner(testf1,testf1)*dx)**0.5
        testf1.vector()[:]=test1/nor
        testf2=Function(V)
        testf2.vector()[:]=test2
        nor=assemble(inner(testf2,testf2)*dx)**0.5
        testf2.vector()[:]=test2/nor
        print w[i],assemble(inner(frx1s,testf1)*dx),assemble(inner(frx2s,testf2)*dx),assemble(inner(frx1s,frx1s)*dx),assemble(inner(frx2s,frx2s)*dx),assemble(inner(testf1,testf1)*dx),assemble(inner(testf2,testf2)*dx)




