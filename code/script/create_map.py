import json as j
import random as r

memory = ".\\media\\test"

def save_matrix(path, matrix):
    """
    save the matrix in the file pointed by the path.
    the path can be relative or absolute.
    """
    try:
        with open(path, "w", encoding="utf-8") as fichier:
            j.dump(matrix, fichier, indent=4)
        return True
    except:
        return False

def calculate_cost(distance, rapidite):
    """
    Calcule le coût d'une route en fonction de la distance et du temps (Exemple simple)
    """
    return distance * 0.5 + rapidite * 0.2

def perturb_value(value, percent=10):
    """
    Return a little bit différent value : [value - percent%, value + percent%]
    """
    variation = value * percent / 100
    return round(r.uniform(value - variation, value + variation), 2)

def create_random_matrix(size):
    """
    Crée une matrice représentant un graphe avec des routes aléatoires
    """
    matrix = [[None for _ in range(size)] for _ in range(size)]

    for i in range(size):
        for j in range(i, size):
            if i == j:
                matrix[i][j] = None
            else:
                distance = r.uniform(5, 100)
                rapidite = r.uniform(5, 120)
                payage = r.randint(1, 20)
                cout = calculate_cost(distance, rapidite)

                road = {
                    "payage": round(payage,2),
                    "distance": round(distance,2),
                    "rapidite": round(rapidite,2),
                    "cout": round(cout,2)
                }

                # Symétrie pour les deux sens
                matrix[i][j] = road

                reversed_road = {
                    "payage": round(payage,2),
                    "distance": perturb_value(round(distance,2)),
                    "rapidite": perturb_value(round(rapidite,2)),
                    "cout": perturb_value(round(cout,2))
                }

                matrix[j][i] = reversed_road.copy()

    return matrix

def create_test(size):
    mat = create_random_matrix(size)
    path = memory + f"\\test_{size}.json"
    statu = save_matrix(path, mat)
    return statu
