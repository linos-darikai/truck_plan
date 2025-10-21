# <center>Truck plan</center>

## Table of Contents

- [I) Introduction](#i-introduction)
  - [I.1. Context](#i1-context)
  - [I.2. Problem Context](#i2-problem-context)
  - [I.3. Problem Statement](#i3-problem-statement)
    - [I.3.a. Basic Constraints](#i3a-basic-constraints)
    - [I.3.b. Additional Constraints](#i3b-additional-constraints)
- [II) Type of Problem](#ii-type-of-problem)
- [III) Mathematical Modeling](#iii-mathematical-modeling)
- [IV) Resolution Method (Algorithmic Approach)](#iv-resolution-method-algorithmic-approach)
- [V) Experimental Study and Results](#v-experimental-study-and-results)
- [VI) Discussion and Limitations](#vi-discussion-and-limitations)
- [VII) Conclusion and Perspectives](#vii-conclusion-and-perspectives)


---

## I) Introduction

### I.1. Context
Since the 1990s, global concern over energy use and greenhouse gas emissions has grown. The 1997 Kyoto Protocol marked the first major international commitment but was soon seen as insufficient. Since then, more ambitious goals—such as France’s target to reduce emissions by 75% by 2050—have emerged. However, governments cannot easily enforce behavioral changes, so efforts focus on promoting energy efficiency, recycling, and sustainable transport.

---

### I.2. Problem Context
ADEME has launched a call for projects to test innovative mobility solutions. CesiCDP, already active in Smart Multimodal Mobility, plans to respond by focusing on optimizing delivery routes. The goal is to develop an **Operations Research** method capable of finding the shortest route connecting selected cities while considering traffic variations over time. Adding realistic constraints could make the proposal more attractive to ADEME, though it would also increase its complexity.

---

### I.3. Problem Statement

#### I.3.a. Basic Constraints
* Solve large-scale instances (several thousand cities).  
* Conduct a statistical study of the algorithm’s experimental performance.

#### I.3.b. Additional Constraints
* Multiple trucks (*k* trucks) available simultaneously

  * Each truck transport a type of package
  * Each truck have a capacity define
  * Each truck have a fuel modifier

* Constant travel time

  * Matrix of variable distances per time slot

---

## II) Type of Problem
### II.1.Mathematical Modeling
#### <u>Optimisation problem:</u>

Input:<br>
* $T \in \mathbb{N}^\mathbb{N}$ list of trucks where each value indicating its class and allowed object types.
* $c$ the constraint chose (distance, duration, price) 
* $G = \left\{ V, E,\omega\right\}$ a graph with <br>
  * $V$ the set of vertices such as <br>
$\forall v ∈ V, v = (t, p, w)$ with
    * $t$ : type of node, either "stock", "shop", or "depot"  
    * $p$ : list of products available or requested at the node  
    * $w$ : time window for the node, represented as (start_time, end_time)
  * $E \in V^2$ the set of edges,
  * $\omega \in (E \times c)$
    * if $c = p$ it return the toll associate to the edge  
    * if $c = dis$ it return distance of the edge
    * if $c = dur$ it return expected duration to traverse the edge <br>
<br>

Output:<br><br>
The list of each truck path in fonction of the constraint ($\in V^{\mathbb N ^{\mathbb N}}$)

#### <u>Decision problem assocaite:</u>

Input:<br>
* $B \in \mathbb{N}^\mathbb{N}$ list of trucks where each value indicating its class and allowed object types.
* $c$ the constraint chose (distance, duration, price)
* $G = \left\{ V, E,\omega\right\}$ a graph with <br>
  * $V$ the set of vertices such as <br>
$\forall v ∈ V, v = (t, p, w)$ with
    * $t$ : type of node, either "stock", "shop", or "depot"  
    * $p$ : list of products available or requested at the node  
    * $w$ : time window for the node, represented as (start_time, end_time)
  * $E \in V^2$ the set of edges,
  * $\omega \in (E \times c)$
    * if $c = p$ it return the toll associate to the edge  
    * if $c = dis$ it return distance of the edge
    * if $c = dur$ it return expected duration to traverse the edge <br>

* $K$ the thresold of our problem
<br>

Output:<br><br>
All the truck are come back
* and the sum of the distance of all bus path is less than $K$
* before the duration $K$ if c is duration
* and paid less than $K$ if c is price 


### II.2.Proof of the dificulties

The Decision problem is NP?

We take a certificate C which is the list of each truck paths.

**DEBUT**</br>
INITIALISATION:</br>
$saw =$ [], $c = constraint$

LOOP START $i=0$ FROM $0$ TO $len(C)$
* $total = 0$
* LOOP START $y=0$ FROM $0$ TO $len(C[i])$
  * CONDITION : $C[i][y]$ not in saw ?
    * YES → Add $C[i][y]$ in saw
    * NO → Pass
  * CONDITION : ($C[i][y]$, $C[i][y+1]$) in E ?
    * YES → $total$ += $\omega((C[i][y], C[i][y+1]),c)$
    * NO → RETURN False
  * CONDITION : C[i][0] == C[i][-1] and total <= k ?
    * YES → $total$ += $\omega((C[i][y], C[i][y+1]),c)$
    * NO → RETURN False
* LOOP END

LOOP END</br>
CONDITION : len(already_saw)-1 == len(V) ?
* Oui → Retourner True
* Non → Retourner False

**FIN**</br></br></br>



---

## III) data format
| Nœud | Type  | Ressources | Horaire d’ouverture | Connexions vers… |
|------|-------|------------|-------------------|-----------------|
| 0    | shop  | wood       | 12:15 – 22:45     | 1 → Toll 8, 38.17 km, 35.54 min, Cost 26.19<br>3 → Toll 6, 47.54 km, 13.37 min, Cost 26.44 |
| 1    | depot | metal, wood| 00:00 – 24:00     | 0 → Toll 8, 41.34 km, 37.35 min, Cost 23.74<br>2 → Toll 18, 36.67 km, 109.65 min, Cost 40.26<br>4 → Toll 12, 92.13 km, 20.09 min, Cost 50.08 |
| 2    | shop  | metal      | 12:00 – 15:15     | 1 → Toll 18, 37.12 km, 102.63 min, Cost 37.06<br>3 → Toll 7, 49.11 km, 67.50 min, Cost 38.06<br>4 → Toll 7, 38.75 km, 39.02 min, Cost 27.18 |
| 3    | stock | honey      | 19:00 – 21:45     | 0 → Toll 6, 46.23 km, 13.96 min, Cost 26.13<br>2 → Toll 7, 45.81 km, 71.91 min, Cost 37.92<br>4 → Toll 11, 22.61 km, 119.57 min, Cost 35.22 |
| 4    | stock | honey      | 09:30 – 15:00     | 1 → Toll 12, 85.67 km, 18.67 min, Cost 47.21<br>2 → Toll 7, 38.56 km, 40.46 min, Cost 28.15<br>3 → Toll 11, 20.78 km, 116.95 min, Cost 35.65 |

The cost is the price of gaz, calculated with distance and the duration.
