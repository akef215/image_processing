import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from bigtree import Node

class RAG:
    """
        Une implementation POO d'un graphe d'adjacence de région d'une image donnée

        Attributs:
            img (ndarray(n, m)) : The image that the RAG is representing
            vertices (list) : The list of vertices of the RAG
            edges (list) : The list of edges of the RAG
            following the format [(vertx_1, vertex_2) / vertex_1, vertex_2 belong to vertices]
            adj_dict (dict) : The adjacency dictionnary to simplify the merge part
            quad_tree (bigtree.Node) : The quad tree representation of img

        The constructor take as arguments just the img, so he can construct first quad_tree and than the RAG    

    """
    # Constructeur:
    def __init__(self, img):
        self.img_ = img
        self.quad_tree_ = None
        self.size_ = 0
        self.edges_ = []
        self.adj_dict_ = {}

    # Constantes de 4NN
    # les indices qui vont etre utilisé pour extraire 
    # les composantes (haut, bas, gauche et droite)
    # des 4 plus proches voisins
  
    up = [0, 1]
    down = [2, 3]
    right = [1, 3]
    left = [0, 2]

    # Méthodes statistiques:
    @staticmethod
    def split_image(img):
        """
            Dévise une image en 4 sous-images suivant le schéma suivant:
            +-----+-----+
            +  0  +  1  +
            +-----+-----+
            +  2  +  3  +
            +-----+-----+

            Args:
                img (ndarray(n, m)) : L'image à diviser
            
            Retourne:
                subimages (list) : Une liste qui contient les 4 sous-images
                , réparties suivant le schéma sus-mentionné                  
        """  
        # r et c représentent le nombre de lignes et nombre de colonne, 
        # respectifs, de chaque sous image résultante      
        r = int(np.floor(img.shape[0]/2))
        c = int(np.floor(img.shape[1]/2))
        subimages = [] 
        for i in range(2):
            for j in range(2):
                subimages.append(img[r*i:r*(i+1), c*j:c*(j+1)])
        return subimages 
    
    @staticmethod
    def homogeneity_split(img, thresh=2):
        """
            Le test d'homogénité utilisé par défaut dans la procédure de division
            Ca consiste a vérifier si l'étendu de l'image ne dépasse pas un seuil donné

            Args:
                img (ndarray(n, m)) : L'image.
                thresh (int, optionnel) : le seuil a ne pas dépasser.
                Valeur par défaut = 2.

            Retourne:
                Un booléan vrai si l'image est homogène, faux sinon.        
        """
        diff = np.max(img) - np.min(img)
        return diff < thresh
    
    @staticmethod
    def homogeneity_merge(img1, img2, thresh):
        """
            Le test d'homogénité utilisé par défaut dans la procédure de fusion
            Ca consiste a vérifier si l'étendu de l'union des deux images
            à fusioner ne dépasse un certain seuil donné

            Args:
                img1 (ndarray(n, m)) : L'image 1.
                img2 (ndarray(n', m')) : L'image 2.
                thresh (int, optionnel) : le seuil a ne pas dépasser.
                Valeur par défaut = 2.

            Retourne:
                Un booléan vrai si l'union des deux images est homogène
                , faux sinon.        
        """
        max_global = max(np.max(img1), np.max(img2))
        min_global = min(np.min(img1), np.min(img2))
        return max_global - min_global < thresh

    def quad_tree(self):
        """
            Construire l'arbre étiqueté par des entiers correspondants
            a un parcours en largeur (Breadth First Traversal)
            
            Args:
                img (ndarray(n, m)) : L'image d'attribut. 
                size (int) : Le nombre de noeuds de l'arbre résultant.    
        """
        def build_quad_tree(img, label="parent", homogeneity_split=self.homogeneity_split):
            """
                La construction récursive de l'arbre de telle sorte
                que chaque noeud comprend soit 0, soit 4 fils seulement
                et où chaque noeud représente une région de l'image img.

                Args:
                   img (ndarray(n, m)) : L'image.

                   label (str, optionnel) = Le nom de la racine. 
                   Par défaut : "parent" 

                   homogeneity_split (func, optionnel) : Le critère 
                   d'homogénité de l'image
                   Par défaut : self.homogeneity_split préalablement définie

                Retourne:
                    root (bigtree.Node) : La racine de l'arbre créé    
            """
            # On construit une racine ayant label pour nom 
            # et img pour valeur 
            root = Node(label, value=img)
            # Si l'image d'entrée ne respecte pas le critère
            # d'homogénité qu'on a imposée elle sera divisée 
            # en 4 images. Or, elle aura 4 fils.
            if not homogeneity_split(img):
                sub = self.split_image(img)
                for i, sub_img in enumerate(sub):
                    # L'appel récursif de la construction des fils
                    child = build_quad_tree(sub_img, label=f"child_{i}")
                    # Association des fils leur racine
                    child.parent = root
            return root
        
        # Construction d'un arbre étiqueté avec parent pour la racine et 
        # child_i pour le reste des noeuds, s'il en existent,
        tree = build_quad_tree(self.img_, "parent")
        # Initiaon de la file d'attente pour effectuer 
        # un parcour en largeur 

        file = deque([tree])
        # Initialisation de la liste de sortie, contenant le DF
        out = []
        # Un compteur id pour étiqueter les noeuds de l'arbre selon
        # l'ordre du DFS
        id = 0
        # Repeter jusqu'à ce que la file devient vide
        while len(file) != 0:
            # Défiler le noeud
            node = file.popleft()
            # Le rajouter à la liste de sortie 
            # changer le nom du noeud par son ordre de DFS
            out.append(node.value)
            node.value = ([id], out[id])
            node.name = str(id)
            id += 1

            # Itereter sur les fils de chaque noeud
            for child in node.children:
                file.append(child)

        self.quad_tree_ = tree
        self.size_ = id
    
    # Graphe_Division
    def format_children(self, node):
        """
            Formater les fils d'un noeud donné, sous forme d'un tableau de 4 
            entiers représentant les noms des noeuds fils, selon leurs ordre DFS.
            Dans le cas d'une feuille, ça retourne un tableau de la forme
            [node.id, node.id, node.id, node.id] afin de faciliter 
            la conversion de l'arbre en RAG

            Args:
                node (bigtree.Node) : Le noeud utilisé.

            Retourne:
                children (list) : liste des noms de fils du node, s'ils en 
                existent, sinon le id du node répété 4 fois.    
        """
        # Vérifier si le noeud n'a pas de fils
        if len(node.children) == 0:
            children = np.ones(shape=(4), dtype=int)*node.value[0][0]
        else:
            vals = []
            # Récupérations des id des noeuds fils
            for child in node.children:
                vals.append(child.value[0][0])
            children = np.array(vals, dtype=int)
        return children

    def edges_from_parent(self, node):
        """
            La liste des arretes suivant le schéma de l'image.
            De telle sorte que (u, v) est une arrete valide ssi u et v
            sont fils du meme noeud et représentent deux regions adjacentes 
            dans l'image (self.img_) 

            NB : 
                La représentation en tuple est pour 
                simplifier le traitement c'est juste un abus de langage.

            Rappel du schéma de division d'une image:
            +-----+-----+
            +  0  -  1  +
            +--|--+--|--+
            +  2  -  3  +
            +-----+-----+

            d'où les seules arretes à prendre en compte sont sous la forme
            {(child[0], child[1]), (child[0], child[2]),
            (child[1], child[3]), (child[2], child[3])}

            Args:
                node (bigtree.Node) : Le noeud utilisé.
            
            Retourne:
                La liste des id de noeuds fils du node respectant les condtions
                discutées ci-dessus. Si node n'a pas de fils ça retourne [] 
        """
        children = node.children
        if len(children) == 0:
            return [] 
        return [(children[0].value[0][0], children[1].value[0][0]),
                    (children[0].value[0][0], children[2].value[0][0]),
                    (children[1].value[0][0], children[3].value[0][0]),
                    (children[2].value[0][0], children[3].value[0][0]),
                    ]
    
    def edges_contains_vertex(self, edges, vertex):
        """
            La liste des arretes ayant vertex comme sommet initial ou final

            Args:
                edges (list) : La liste des arretes, le domaine de recherche.
                vertex (int) : Le sommet qui doit etre inclus, critère de recherche.
            
                Retourne:
                    contains (list) : La liste des arretes comprenant de vertex
        """
        return [e for e in edges if vertex in e]
    
    def get_node(self, root, name="0"):
        """
            Recherche récursive d'un nœud par son nom.  
        """
        if root.name == name:
            return root
        for child in root.children:
            # Appel récursif de get_node pour explorer le reste 
            # de l'arborescence
            result = self.get_node(child, name)
            if result is not None:
                return result
        return None
    
    def get_sub_image_at(self, i):
        """
            Retourne l'image associé au noeud d'id i
        """
        return self.get_node(self.quad_tree_, str(i)).value[1]
    
    def profondeur(self, node):
        """
            La profondeur d'un noeud
        """
        p = 0
        while node.parent is not None:
            p += 1
            node = node.parent
        return p
    
    def parent_at_level(self, node, level=1):
        """
            Retrouver le predecesseur du node se trouvant au level indiqué

            Args:
                node (bigtree.Node) : le noeud d'origine
                level (int) : le niveau dans lequel se trouve l'ancetre
                qu'on cherche
            
            Retourne:
                L'id (int) du predecesseur du node au niveau level
        """
        if self.profondeur(node) < level:
            return -1
        while (self.profondeur(node) > level):
            node = node.ancestors.__next__() 

        return node.value[0][0]
    
    def build_edges(self, edges):
        """
            Quand on divise un noeud en 4 fils
            il suivra le schéma suivant : 
            +-------+-------+
            +   n   +  n+1  +
            +-------+-------+
            +  n+1  +  n+2  +
            +-------+-------+
            c'est du à l'étiquetage par ordre DFS

            Donc pour qu'il y ait une adjacence horizentale
            il faut que la difference des id soit égal à 1
            et pour une adjacence vertical il faut que cette différence
            soit égale à 2

            Il faut observer aussi le faite que le min(id) se trouvera toujours 
            à gauche (en haut) et le max(id) se trouvera à droite (en bas)

            On va se servir de cette observation pour établir le lien d'ajdancence

            Dans le cas des noeuds originaire de multiple divisions de noeuds
            , eventuellement faisant partie de différent niveau on fixe comme
            repère le niveau 1 (1, 2, 3, 4) qui respectent bien notre
            derniere observation tout en assurant une sorte de normalisation. 

            On rappelle que self.format_children retourne les fils sous
            forme d'une liste de 4 element, ou le id du noeud répété 4 fois
            s'il n'a pas de fils et ce pour pouvoir établir le lien d'adjacence
            par exemple 
            +-----+-----+
            + 1 1 + 5 6 +
            + 1 1 + 7 8 +
            +-----+-----+
            le noeud 1 est adjacent 'de gauche' des noeuds 5 et 7
            Avec la méthode format_children on peut établir ce lien
            et faire correspondre par exemple le 1 à plusieurs noeuds
            et ce à l'aide des masque de positions travaillant 
            à la base du schéma suivant:
             left | right
            +-----+-----+
            +  0  +  1  + Up
            +-----+-----+
            +  2  +  3  + Down
            +-----+-----+
        """
        def diff(edge):
            # edge c'est l'arrete d'entrée (edge[0], edge[1])
            e1 = self.get_node(self.quad_tree_, str(edge[0]))
            e2 = self.get_node(self.quad_tree_, str(edge[1]))
            # Retourne la différence absolue normalisée, selon la définition
            # donnée dans la documentation de la méthode par rapport au niveau1
            return abs(self.parent_at_level(e1, 1) \
                    - self.parent_at_level(e2, 1))
        
        out = []
        
        for edge in edges:
            # si la différence vaut 0 alors il s'agit d'adjacence verticale
            # sinon (diff == 1) c'est une adjacence horizentale

            edge_max = self.get_node(self.quad_tree_, str(max(edge)))
            edge_min = self.get_node(self.quad_tree_, str(min(edge)))
            if diff(edge) % 2 == 0:
                haut = self.format_children(edge_max)[self.up]
                bas = self.format_children(edge_min)[self.down]
                # Construire les nouvelles arretes
                out.extend([(int(haut[i]), int(bas[i])) for i in range(2)])
            else:
                gauche = self.format_children(edge_max)[self.left]
                droite = self.format_children(edge_min)[self.right]
                # Construire les nouvelles arretes
                out.extend([(int(droite[i]), int(gauche[i])) for i in range(2)])    
        return out
       
    def build_RAG(self):
        """
            Construction du graphe d'adjacence de regions (RAG)
            et mise à jours des champs liés aux graphes
        """
        edges_g = []
        def recursive_init(node):
            # Ajout des cycles connexes du meme parent
            edges_g.extend(self.edges_from_parent(node))
            for child in node.children:
                # appel récursif pour couvrir tout l'arbre
                recursive_init(child)

        recursive_init(self.quad_tree_)
            
        # supprimer les doublons, s'ils existent
        edges_g = list(set(edges_g))
        # edge_g contient initialement tous les cycles connexes
        # dus au multiples divisions
        for i in range(self.size_):
            # récuperer les noeuds de l'arbre un par un selon leur ordre de BFS
            child = self.get_node(self.quad_tree_, str(i))
            # si le noeud n'est pas une feuille
            if (child.children != ()):
                # Trouver toutes les arretes composées de ce child
                edges_vertex = self.edges_contains_vertex(edges_g, child.value[0][0])  
                # Construire les arretes d'adjacences entre ces cycles     
                edges_g.extend(self.build_edges(edges_vertex))
                # Enlever les arretes liées à des sommets qui ont subi 
                # une division
                edges_g = list(set(edges_g) - set(edges_vertex))
        
        self.edges_ = edges_g
        self.adj_dict_ = {i : [] for i in range(self.size_)}
        for edge in self.edges_:
            self.adj_dict_[edge[0]].append(edge[1])
            self.adj_dict_[edge[1]].append(edge[0]) 

    # Graphe_Fusion

    # Méthodes d'affichage
    def plot_graph(self):
        """
        Affiche un graphe à partir d'une liste d'arêtes.

        edges : list of tuples
            Liste des arêtes, par exemple [(0,1),(1,2),(2,3)]
        labels : dict (optionnel)
            Dictionnaire {node_id: label} pour afficher des labels personnalisés
        """
        # Création du graphe
        G = nx.Graph()
        G.add_edges_from(self.edges_)
        
        # Positionnement des nœuds (force-directed layout)
        pos = nx.spring_layout(G, seed=42)
        
        # Dessin des nœuds
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        
        # Dessin des arêtes
        nx.draw_networkx_edges(G, pos, width=2)
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
        plt.title("Graphe d'adjacence de region de l'image")
        plt.axis('off')
        plt.show()
    
    def print_quad_tree(self):
        print("L'arbre de la division de l'image :")
        self.quad_tree_.hshow()

    # The getters:
    def get_quad_tree(self):
        return self.quad_tree_

    def get_edges(self):
        return self.edges_

    def get_adj_dict(self):
        return self.adj_dict_ 

    def get_tree_size(self):
        return self.size_  