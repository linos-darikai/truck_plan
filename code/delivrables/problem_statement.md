# <center>Truck plan</center>

## Table of Contents

* [I) Introduction](#i-introduction)
  * [I.1. Context](#i1-context)
  * [I.2. Problem Context](#i2-problem-context)
  * [I.3. Problem Statement](#i3-problem-statement)
    * [I.3.a. Basic Constraints](#i3a-basic-constraints)
    * [I.3.b. Additional Constraints](#i3b-additional-constraints)
* [II) Type of Problem](#ii-type-of-problem)
  * [II.1. Mathematical Modeling](#ii1-mathematical-modeling)
  * [II.2. Proof of NP membership of the problems](#ii2-proof-of-np-membership-of-the-problems)
    * [II.2.a. The decision problem is NP?](#ii2a-the-decision-problem-is-np)
    * [II.2.b. The optimisation problem is NP?](#ii2b-the-optimisation-problem-is-np)
  * [II.3. Proof of NP-Hard membership of the problems](#ii3-proof-of-np-hard-membership-of-the-problems)
    * [II.3.a. The decision problem is NP-Complete?](#ii3a-the-decision-problem-is-np-complete)
    * [II.2.b. The optimisation problem is NP-Hard?](#ii2b-the-optimisation-problem-is-np-hard)
* [III) Mathematical Modeling](#iii-mathematical-modeling)



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
### II.1. Mathematical Modeling
#### <u>Optimisation problem:</u>

Input:
* $T \in (\mathbb{N}^\mathbb{N},\mathbb{N},\mathbb{N})^\mathbb{N}$ List of trucks as (Carriable objects, max volume, max weight)
* $G = \left\{ V, E,\omega,\phi\right\}$ a graph with <br>
  * $V$ the set of vertices,
  * $E \in V^2$ the set of edges,
  * $\omega : E \times \mathbb{N} \rightarrow \mathbb{N}^\mathbb{N}$ a function returning time-variation polynomial coefficients.
  * $\phi: V \to \mathbb{N}^\mathbb{E}$, with $\mathbb{E}$ the set of sellable products — returns shop needs.
<br>

Output:<br><br>
Let $VL$ be the set of feasible lists of cycles such that
$$
\forall v \in V, \forall e \in \mathbb{E}, \quad \phi(v)[e] = \sum_{n=0}^{|T|} l[n][1][e].
$$
Let $l^*$ such as 
$$
l^* =\min_{l \in VL} \max_n |l[n]|.
$$
Return the element $l^*$ 

#### <u>Decision problem assocaite:</u>

Input:
* $T \in (\mathbb{N}^\mathbb{N},\mathbb{N},\mathbb{N})^\mathbb{N}$ List of trucks as (Carriable objects, max volume, max weight)
* $G = \left\{ V, E,\omega,\phi\right\}$ a graph with <br>
  * $V$ the set of vertices,
  * $E \in V^2$ the set of edges,
  * $\omega : E \times \mathbb{N} \rightarrow \mathbb{N}^\mathbb{N}$ a function returning time-variation polynomial coefficients.
  * $\phi: V \to \mathbb{N}^\mathbb{E}$, with $\mathbb{E}$ the set of sellable products — returns shop needs.
* $K$ the thresold of our problem

Output:<br><br>
Return if it exist a list $l$ ($l \in (V, \mathbb{N}^{E})^{\mathbb{N}^{\mathbb{N}}}$) such as
$$\forall v \in V, \forall e \in \mathbb{E}, \phi(v)[e] = \sum_{n \in [0,|T|]} l[n][1][e]$$
$$ \max_n |l[n]| \leq K$$
<br>

---

### II.2. Proof of NP membership of the problems

#### II.2.a. The decision problem is NP?
We take a certificate C which is the list of each truck paths.
<pre>
INITIALISATION :
delivery = [0] × len(V) × len(ℰ)
time = []

LOOP START i FROM 0 TO len(C)
    total = 0
    max_volume = T[i][1]
    max_weight = T[i][2]

    LOOP START y FROM 0 TO len(C[i])
        CONDITION : (C[i][0][y], C[i][0][y+1]) ∈ E ?
            YES →
                total += ω((C[i][0][y], C[i][0][y+1]), total)
                LOOP START z FROM 0 TO len(C[i][1])
                        CONDITION : z ∉ T[i][0] ?
                            YES → RETURN False
                            NO → PASS
                    delivery[y][z] += C[i][1][z]
                    max_volume -= C[i][1][z] * z(0)
                    max_weight -= C[i][1][z] * z(1)
                LOOP END
            NO → RETURN False

        CONDITION : C[i][1][0] ≠ C[i][1][-1] ?
            YES → RETURN False
            NO → PASS

    LOOP END

    CONDITION : max_volume ≤ 0 and max_weight ≤ 0 ?
        YES → RETURN False
        NO → PASS

    time.APPEND(total)

LOOP END

CONDITION : max(time) ≤ K ?
    YES → RETURN False
    NO → PASS

LOOP START v FROM 0 TO len(V)
    LOOP START n FROM 0 TO len(ℰ)
        CONDITION : delivery[v][n] ≠ φ(V)[n] ?
            YES → RETURN False
            NO → PASS
    LOOP END
LOOP END

RETURN True
</pre>


This algorigram verify:
* All trucks complete a **full cycle**:  
  * They do not give more than they transport.  
  * They only carry what they are able to.
* All shops receive **exactly the products** they need.
* All deliveries are completed in **less than or equal to \(K\) time**.

Moreover, the algorithm has an asymptotic complexity of $$O(|T| \times |V| \times |\mathbb{E}|)$$

Therefore, this algorithm verifies whether a given certificate is a valid solution to the problem in polynomial time.

Hence, this problem belongs to the class NP.


#### II.2.b. The optimisation problem is NP?
We take a certificate C which is the list of each truck paths.

to prove $C =\min_{l \in VL} \max_n |l[n]|$ you need to calculate all the l in Vl so you need to calculate all the potential solution and it's not calculable in polynomial time.

So the optimisation problem is not in NP.

---

### II.3. Proof of NP-Hard membership of the problems

#### II.3.a. The decision problem is NP-Complete?


#### II.2.b. The optimisation problem is NP-Hard?
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
