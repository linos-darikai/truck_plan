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
* [III) Data format](#iii-data-format)
* [IV) What are the next steps?](#iv-what-are-the-next-steps)
* [V) Sources](#v-sources)



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
#### <u>Optimisation problem ($\Pi_O$):</u>

Input:
* $T \in (\mathbb{N}^\mathbb{N},\mathbb{N},\mathbb{N})^\mathbb{N}$ List of trucks as (Carriable objects, max volume, max weight)
* $\mathbb{E} \in (\mathbb{N}^2)^\mathbb{N}$ the list of sellable products as (unit volume, unit weight)
* $G = \left\{ V, E,\omega,\phi\right\}$ a graph with <br>
  * $V$ the set of vertices,
  * $E \in V^2$ the set of edges,
  * $\omega : E \times \mathbb{N} \rightarrow \mathbb{N}^\mathbb{N}$ a function returning time-variation polynomial coefficients.
  * $\phi: V \to \mathbb{N}^{|\mathbb{E}|}$ returns shop needs.

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

#### <u>Decision problem assocaite ($\Pi_D$):</u>

Input:
* $T \in (\mathbb{N}^\mathbb{N},\mathbb{N},\mathbb{N})^\mathbb{N}$ List of trucks as (Carriable objects, max volume, max weight)
* $\mathbb{E} \in (\mathbb{N}^2)^\mathbb{N}$ the list of sellable products as (unit volume, unit weight)
* $G = \left\{ V, E,\omega,\phi\right\}$ a graph with <br>
  * $V$ the set of vertices,
  * $E \in V^2$ the set of edges,
  * $\omega : E \times \mathbb{N} \rightarrow \mathbb{N}^\mathbb{N}$ a function returning time-variation polynomial coefficients.
  * $\phi: V \to \mathbb{N}^\mathbb{E}$ returns shop needs.
* $K$ the threshold of our problem

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

to prove $C =\min_{l \in VL} \max_n |l[n]|$ you need to calculate all the $l$ in $VL$ so you need to calculate all the potential solution and it's not calculable in polynomial time.

So the optimisation problem is not in NP.

---

### II.3. Proof of NP-Hard membership of the problems
#### II.3.a. The decision problem is NP-Complete?
##### remind:
- The Hamiltonian cycle decision problem is in NP-Complete so it's also in NP.

</br>
The Hamiltonian cycle decision problem is:</br>
Input:

* $H = \left\{ V_H, E_H\right\}$ a graph with
  * $V_H$ the set of vertices,
  * $E_H \in {V_H}^2$ the set of edges,
<br>

Output:<br>
 * Return if we can find an Hamiltonian cycle in G.<br><br>

##### demonstration:


We construct, in polynomial time, an instance $I = (T,\mathbb{E},G,K)$ of $\Pi_D$ such that

$$
\text{H has a Hamiltonian cycle} \Leftrightarrow
\text{I has a feasible solution}
$$
Let:
$$
\begin{array}{rcl}
T &=&[[0],|V_H|,|V_H|]\\
\mathbb{E} &=&[(1,1)]\\
G &=&\left\{ V, E,\omega,\phi\right\} \text{ with}\\
&& V = V_H \\
&& E = E_H \\
&& \omega = 1 \\
&& \phi = [1] \\
K &=&|V_H|\\
\end{array}
$$

with simple words:
* each vertex requires exactly one product,
* the truck can carry all the products at once,
* and each edge has a constant travel time of 1

</br>

$(\Rightarrow)$ If $H$ has a Hamiltonian cycle, then 
$I$ has a valid solution</br>

Let $ C_H = (v_1, v_2, \dots, v_{|V|}, v_1) $ be a Hamiltonian cycle in $ H $.

Since $ E = E_H $, this cycle is also feasible in $ G $.
Let the truck follow exactly this route.

At each visited vertex $ v_i $, it delivers the single required product:
$ \phi(v_i) = [1] $.

The total length of the tour is:
$ |C_H| = |V| $.

Since $ K = |V| $, we have:
$ \max_n |l[n]| \leq K $.

Thus, a feasible delivery list $ l $ exists for $ I $.

---

$(\Leftarrow)$ If $ I $ has a feasible solution, then $ H $ has a Hamiltonian cycle.

Suppose that $ I $ admits a feasible solution
$ l = [l_1] $,
since there is only one truck.

Then, for each vertex $ v \in V $, the delivery constraint:
$ \forall v \in V, \quad \phi(v) = \sum_n l_n[1] $
implies that each vertex receives one product.

Since the truck can carry all the items and the travel cost of each edge is $ 1 $,
the route $ l_1[0] $ must visit all vertices exactly once in order to satisfy all deliveries without exceeding
$ K = |V| $.

Therefore,
$ l_1[0] $
corresponds to a Hamiltonian cycle in $ H $.

So $\Pi_D$ is in NP-Complete.

---

#### II.2.b. The optimisation problem is NP-Hard?

##### remind:
- The Hamiltonian cycle decision problem is in NP-Complete so it's also in NP.

</br>
The Hamiltonian cycle decision problem is:</br>
Input:

* $H = \left\{ V_H, E_H\right\}$ a graph with
  * $V_H$ the set of vertices,
  * $E_H \in {V_H}^2$ the set of edges,
<br>

Output:<br>
 * Return if we can find an Hamiltonian cycle in G.

##### demonstration:
It’s almost the same proof as $\Pi_D$.<br>
We construct, in polynomial time, an instance $I = (T,\mathbb{E},G)$ of $\Pi_O$ such that

$$
\text{H has a Hamiltonian cycle} \Leftrightarrow
\text{I has a feasible solution}
$$
Let:
$$
\begin{array}{rcl}
T &=&[[0],|V_H|,|V_H|]\\
\mathbb{E} &=&[(1,1)]\\
G &=&\left\{ V, E,\omega,\phi\right\} \text{ with}\\
&& V = V_H \\
&& E = E_H \\
&& \omega = 1 \\
&& \phi = [1] \\
\end{array}
$$

with simple words:
* each vertex requires exactly one product,
* the truck can carry all the products at once,
* and each edge has a constant travel time of 1


$(\Rightarrow)$ If $H$ has a Hamiltonian cycle, the solution $l$ of $I$ in $\Pi_O$ respect $l[0] = |V|$</br>

Let $ C_H = (v_1, v_2, \dots, v_{|V|}, v_1) $ be a Hamiltonian cycle in $ H $.

Since $ E = E_H $, this cycle is also feasible in $ G $.
Let the truck follow exactly this route.

At each visited vertex $ v_i $, it delivers the single required product:
$ \phi(v_i) = [1] $.

The total length of the tour is:
$ |C_H| = |V| $.

But as The truck must visit every vertex $v \in V$ at least once to satisfy $\phi (v) = [1]$.

So the lenght of the cycle is a least $|V|+1$.

Therefore, the number of edges used is at least $|V|$, and this number is equal to the weight of the path (since each edge has a weight of 1).

So $ |C_H| \geq |V|$

The condition is respected.

$(\Leftarrow)$ If the solution $l$ of $I$ in $\Pi_O$ respect $l[0] = |V|$, $H$ has a Hamiltonian cycle.

Suppose that $ I $ admits a feasible solution
$ l = [l_1] $,
since there is only one truck.

since, for each vertices v ∈ V, the delivery constraint:
$$\forall v \in V, \phi (v) = 1$$

that means that each vertices receive exactly 1 product.

As the cost of each edge is 1 and the truck can transport all the stuff at once.

So the lenght of the cycle is a least $|V|+1$.

Therefore, the number of edges used is at least $|V|$, and this number is equal to the weight of the path (since each edge has a weight of 1).

So if the length of $l[0] = |V|$, it exist an hamiltonian cycle in $H$

To conclude $\Pi_O$ is in NP-Hard.


Therefore,
$ l_1[0] $
corresponds to a Hamiltonian cycle in $ H $.

So $\Pi_O$ is in NP-Hard.

---
## III) data format

As seen in the mathematical problem modelisation, we need to have acces to 3 datas.

1. **List of `Truck` objects**  
   Each truck has the following attributes:  
   - `truck_type` (`str`): The name/type of the truck.  
   - `allowed_products` (`set` of `str`): Names of products the truck is allowed to carry.  
   - `max_volume` (`int`): Maximum volume the truck can carry.  
   - `max_weight` (`int`): Maximum weight the truck can carry.  
   - `cargo` (`dict` of `Product` → `int`): Dictionary mapping products to their quantities in the truck.

2. **List of `Product` objects**  
   Each product has the following attributes:  
   - `name` (`str`): Name of the product.  
   - `volume` (`int`): Volume of one unit of the product.  
   - `weight` (`int`): Weight of one unit of the product.
   - `delivery_time` (`float`): the time necessary to delivered the product

3. **Graph `G`**  
   Represented as a list of dictionaries:  
   - `G[i][j] = f(t)` is a function returning the time-dependent weight from node `i` to node `j`.  
   - If there is no edge from `i` to `j`, `j` is not a key in `G[i]`.

4. **Solution `S`**  
   Represented as a list of lists of tuple `(node_index, products_delivered, leaving_time)`:  
   - Each outer list corresponds to a truck.  
   - Each inner list represents the sequence of nodes visited by that truck.  
   - Each node contains:  
     - `node_index` (`int`): index of the node visited.  
     - `products_delivered` (`dict` of product identifiers and his quantity): products delivered at that node.
     - `leaving_time` (`int`): the moment when the truck leave the place 

---
## IV) How to solve the problem
### IV.1 Exact solution
Finding an exact solution to this truck routing and delivery problem is computationally infeasible for realistic instances. The problem is NP-Hard, meaning that the time required to compute the optimal solution grows exponentially with the number of trucks, products, and nodes.

### IV.2 Approximate solution


Metaheuristic

Tabu Search: Use memory of visited solutions to avoid cycling and explore better solutions.

Advantages:

Can handle multiple trucks, capacity constraints, and time-dependent travel times.

Flexible: You can add constraints like truck types, product restrictions, and fuel modifiers.

Provides good approximate solutions quickly, even if not optimal.


## What are the next steps?

* search the good way to solve the problem
* implement it
* make a fonction which verify if the solution of the algorythm can be considered or not.
* Performance evaluation
---
## V) Sources
##### Theoretical References:

[Hamiltonian cycle is in NP demonstration](#https://cs.indstate.edu/~bdhome/HamCycle.pdf)


##### Programing library:

[Collection namedtuple](#https://docs.python.org/3/library/collections.html#collections.namedtuple)

[NetworkX](#https://networkx.org/documentation/stable/tutorial.html)

[Pickle](#https://docs.python.org/3/library/pickle.html)