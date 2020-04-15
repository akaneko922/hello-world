#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:38:55 2019

@author: kanekoakihiro
"""

import numpy as np
import random
import functools
import operator
import math

def f(p):
    if random.random() < p:
        return 1
    else:
        return 0

def make_Ai(n,ph):
    Aitr=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            Aitr[i][j]=f(ph)
    return Aitr+Aitr.T

def ichi(aa,bb):
    if aa == bb:
        return 1
    else:
        return 0

def ichi2(aa,bb,cc,dd):
    if aa==bb and cc==dd:
        return 1
    else:
        return 0
"""
Amini(2013)のPseudo-likelihood methods
"""

def plike(A,e):
    n=len(A)
    kl=list(set(e))
    #print(kl)
    klen=len(kl)
    #print(klen)
    """
    ここからの手順をT回繰り返す.
    """
    for s in range(10):
        b = [[0] * klen for i in range(n)]
        for i in range(n):
            for k in range(klen):    
                for j in range(n):
                    b[i][k]=b[i][k]+A[i][j]*ichi(e[j],kl[k])
        #print(b)
        nk=[0] * klen
    
        for k in range(klen):
            for i in range(n):
                nk[k]=nk[k]+ichi(e[i],kl[k])
        #print(nk)
        nkl=[[0] * klen for i in range(klen)]
    
        for k in range(klen):
            for l in range(klen):
                if k!=l:
                    nkl[k][l]=nk[k]*nk[l]
                else:
                    nkl[k][l]=nk[k]*(nk[k]-1)
        #print(nkl)
        Okl=[[0] * klen for i in range(klen)]
    
        for k in range(klen):
            for l in range(klen):
                for i in range(n):
                    for j in range(n):
                        Okl[k][l]=Okl[k][l]+A[i][j]*ichi2(e[i],k,e[j],l)
        #print(Okl)  
        pihatl=[0] * klen
        for l in range(klen):
            pihatl[l]=nk[l]/n
            #print(pihatl)
        Rhat=np.diag(pihatl)
        #print(Rhat)
        Phatlk=[[0] * klen for i in range(klen)]
        for i in range(klen):
            for j in range(klen):
                Phatlk[i][j]=Okl[i][j]/nkl[i][j]
        Phat=np.array(Phatlk)
        #print(Phat)
        lambdahatlk=[[0] * klen for i in range(klen)]
        for i in range(klen):
            for j in range(klen):
                lambdahatlk[i][j]=n*np.dot(Rhat[i], Phat.T[j])
        #print(lambdahatlk)
    
        """
        ここから下を繰り返す.
        """
        N2=10
        count2=1
        while True:
            pihatil=[[0] * n for i in range(klen)]
            for l in range(klen):
                for i in range(n):
                    prodlistl=[]
                    for m in range(klen):
                        if lambdahatlk[l][m] == 0:
                            #print("Stop")
                            break
                        
                        prodlistl.append(math.exp(b[i][m]*math.log(lambdahatlk[l][m])-lambdahatlk[l][m]))
                        bunsi = pihatl[l]*functools.reduce(operator.mul, prodlistl)
                    bunbo=0
                    for k in range(klen):
                        prodlistk=[]
                        for m in range(klen):
                            if lambdahatlk[k][m] == 0:
                                #print("Stop")
                                break
                            prodlistk.append(math.exp(b[i][m]*math.log(lambdahatlk[k][m])-lambdahatlk[k][m]))
                        #print(functools.reduce(operator.mul, prodlistk))
                        plik = functools.reduce(operator.mul, prodlistk)
                        bunbo=bunbo+pihatl[k]*plik
                    pihatil[l][i]=bunsi/bunbo
    
            pihatl=[0] * klen
            for l in range(klen):
                for i in range(n):
                    pihatl[l]=pihatl[l]+pihatil[l][i]
    
            lambdahatlk=[[0] * klen for i in range(klen)]
            for l in range(klen):
                for k in range(klen):
                    pisum=0
                    for i in range(n):
                        lambdahatlk[k][l]=lambdahatlk[k][l]+b[i][k]*pihatil[l][i]
                        pisum=pisum+pihatil[l][i]
                    lambdahatlk[k][l]=lambdahatlk[k][l]/pisum
            count2 += 1
            if count2>N2:
                break
            
        maxlist=[0] * n
        pihatilt=np.array(pihatil).T
    
        for i in range(i):
            maxlist[i]=max(pihatilt[i])
        #print(maxlist)
        for i in range(n):
            e[i]=np.argmax(pihatilt[i])
        #print(e)
    return e

def devG(A,e):
    kl=list(set(e))
    klen=len(kl)
    etl=[]
    for j in range(klen):
        etl.append([i for i, x in enumerate(e) if x == j])

    A_res=[]
    for k in range(klen):
        A1=[]
        for i in etl[k]:
            for j in etl[k]:
                A1.append(A[i][j])
        arr_A1=np.array(A1)
        A2=arr_A1.reshape([len(etl[k]),len(etl[k])])
        A_res.append(A2)
    return A_res