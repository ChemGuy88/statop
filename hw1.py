{\rtf1\ansi\ansicpg1252\paperw11900\paperh16840\margl1440\margr1440\margt1440\margb1440\vieww18540\viewh17280\viewscale100\lin0\rin0
{\papercolor16777215}
{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;\f1\fnil\fcharset0 Helvetica;}
{\colortbl\red0\green0\blue0;\red255\green255\blue255;}
\pard
{\*\pageHeader Helvetica::12px::right::black::0.5::NO::Header Author: Zoe Appleseed [date] - Page [pagenumber] of [totalpages]}
{\*\pageFooter Helvetica::12px::right::black::0.5::NO::Footer Author: Zoe Appleseed [title] - Page [pagenumber] of [totalpages]}
{\*\background {\shp{\*\shpinst\shpleft0\shptop0\shpright0\shpbottom0\shpfhdr0\shpbxmargin\shpbymargin\shpwr0\shpwrk0\shpfblwtxt1\shpz0\shplid1025{\sp{\sn shapeType}{\sv 1}}{\sp{\sn fFlipH}{\sv 0}}{\sp{\sn fFlipV}{\sv 0}}{\sp{\sn fillColor}{\sv 16777215}}{\sp{\sn fFilled}{\sv 1}}{\sp{\sn lineWidth}{\sv 0}}{\sp{\sn fLine}{\sv 0}}{\sp{\sn bWMode}{\sv 9}}{\sp{\sn fBackground}{\sv 1}}}}}
\fi357
\f0\fs24 \cf0  \f1 #!/usr/bin/env python3# -*- coding: utf-8 -*-'''    hw1'''import csv, json, re, requests, string, sysimport matplotlib.pyplot as pltimport numpy as npimport pandas as pdimport scipy as spimport scipy.spatial as spt# hw1.1n = 5m = 5r = 2Y = np.random.rand(n*m)Y = Y.reshape((n,m))X = np.random.rand(n*r)X = X.reshape((n,r))M = Y.T.dot(X)U, d, V = sp.linalg.svd(M, full_matrices=False)D = np.diag(d)m2 = np.hstack((M,np.zeros((n,m-r))))u1, d1, v1 = sp.linalg.svd(m2, full_matrices=False)t = u1.dot(v1.T)# t = u1.T.dot(v1)error = np.linalg.norm(Y-X.dot(t[:,:r].T))print(error)\
}