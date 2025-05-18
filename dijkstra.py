import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

labels = list("12345678")
routers = 8
path = 'C:\\Users\\Andy\\Desktop\\Network.csv' # modificar según sea necesario

# Load csv info
data = pd.read_csv(path, sep=';', header=None, dtype=str)
data = data.replace("np.inf", np.inf).astype(float)
data.index = labels
data.columns = labels

# General data
nodos = ['1', '2', '3', '4', '5', '6', '7', '8']
nodos_analizar = ['1', '2', '3', '4', '5', '6', '7', '8']
nodos_analizados = []
costos = np.ones((routers)) * np.inf
prev_hop = np.zeros((routers))

# Tratamiento para nodo inicial (Inicialización)
nodo_analizar = input('¿Cuál es el nodo que desea analizar?: ')
nodos_analizados.append(nodo_analizar)
nodos_analizar.remove(nodo_analizar)
costos = data.loc[nodo_analizar].to_numpy()
for i in range(routers):
    if costos[i] != np.inf:
        prev_hop[i] = int(nodo_analizar)
# Loop
while(sorted(nodos_analizados) != nodos):
    min_cost = np.inf
    nodo_menor = None

    # Encontrar w no perteneciente a N' tal que D(w) sea mínima
    for nodo in nodos_analizar:
        current_cost = costos[ord(nodo) - ord('1')]
        if current_cost < min_cost:
            min_cost = current_cost
            nodo_menor = nodo

    # Añadir W a N'
    nodo_analizar = nodo_menor
    nodos_analizados.append(nodo_analizar)
    nodos_analizar.remove(nodo_analizar)

    # Actualizar D(v) para cada vecino de w que no pertenezca a N'
    costos_nodo_analizar = data.loc[nodo_analizar].to_numpy() + min_cost
    for nodo in nodos_analizados:
        costos_nodo_analizar[ord(nodo) - ord('1')] = np.inf

    for i in range(routers):
        if costos_nodo_analizar[i] < costos[i]:
            costos[i] = costos_nodo_analizar[i]
            prev_hop[i] = ord(nodo_analizar) - ord('1') + 1

# Impresión de resultados
tabla_rutas = pd.DataFrame({
    'Destino':    labels,
    'Coste':      costos,
    'Siguiente': [labels[int(h)-1] if h != 0 else '-' for h in prev_hop]
})

print("Tabla de enrutamiento resultante:")
print(tabla_rutas.to_string(index=False))


## Gráfico resultante (Powered by AI)
# Crear grafo
G = nx.DiGraph()

# Añadir nodos
for label in labels:
    G.add_node(label)

# Añadir todas las aristas desde el archivo CSV
for i in labels:
    for j in labels:
        if i != j and data.loc[i, j] != np.inf:
            G.add_edge(i, j, weight=data.loc[i, j])

# Obtener nodo de origen
origen = input('¿Cuál es el nodo de origen para la visualización?: ')

# Crear diccionario para mapear rutas desde origen a cada destino
rutas = {}
for dest in labels:
    if dest != origen:
        # Reconstruir la ruta desde prev_hop
        ruta = []
        nodo_actual = dest
        
        # Si no hay ruta disponible, continuar al siguiente destino
        if prev_hop[ord(dest) - ord('1')] == 0:
            continue
            
        # Reconstruir la ruta en reversa siguiendo prev_hop
        while nodo_actual != origen:
            ruta.append(nodo_actual)
            indice_prev = int(prev_hop[ord(nodo_actual) - ord('1')]) - 1
            nodo_actual = labels[indice_prev]
        
        # Añadir el origen y revertir para obtener la ruta correcta
        ruta.append(origen)
        ruta.reverse()
        
        # Guardar la ruta
        rutas[dest] = ruta

# Visualizar la red
plt.figure(figsize=(10, 8))

# Posición de los nodos en círculo
pos = nx.circular_layout(G)

# Dibujar todos los nodos y enlaces con color gris claro (fondo)
nx.draw_networkx_nodes(G, pos, node_color='lightgrey', node_size=500)
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, arrows=True)
nx.draw_networkx_labels(G, pos, font_size=12)

# Destacar el nodo de origen
nx.draw_networkx_nodes(G, pos, nodelist=[origen], node_color='red', node_size=700)

# Dibujar solo las rutas que se van a seguir según el algoritmo
colores = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan']
color_idx = 0

# Añadir líneas más gruesas para las rutas seleccionadas
for destino, ruta in rutas.items():
    # Obtener las aristas de la ruta
    aristas = [(ruta[i], ruta[i+1]) for i in range(len(ruta)-1)]
    
    # Dibujar las aristas de la ruta con un color distinto
    nx.draw_networkx_edges(G, pos, edgelist=aristas, 
                          width=2.5, 
                          alpha=0.8,
                          arrows=True,
                          edge_color=colores[color_idx % len(colores)])
    
    # Resaltar el nodo destino
    nx.draw_networkx_nodes(G, pos, nodelist=[destino], 
                          node_color=colores[color_idx % len(colores)], 
                          node_size=600)
    
    # Incrementar índice de color para el próximo destino
    color_idx += 1

# Añadir etiquetas de peso a todas las aristas
edge_labels = {(i, j): f"{G[i][j]['weight']:.1f}" for i, j in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title(f"Rutas desde el nodo {origen} según algoritmo de Dijkstra")
plt.axis('off')

# Añadir leyenda con las rutas
plt.figtext(0.5, 0.01, "Rutas calculadas:", ha="center", fontsize=12)
y_pos = 0.005
for destino, ruta in rutas.items():
    y_pos -= 0.03
    ruta_str = " → ".join(ruta)
    costo = costos[ord(destino) - ord('1')]
    plt.figtext(0.5, y_pos, f"{origen} a {destino}: {ruta_str} (Costo: {costo:.1f})", 
                ha="center", fontsize=10, 
                color=colores[(list(rutas.keys()).index(destino)) % len(colores)])

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.show()


# Construir grafo solo con los caminos mínimos
G_tree = nx.DiGraph()           # grafo dirigido para respetar sentido “prev→dest”
G_tree.add_nodes_from(labels)

# Añadir únicamente las aristas que efectivamente usa Dijkstra
for idx, h in enumerate(prev_hop):
    if h != 0:
        src = str(int(h))           # nodo previo
        dst = labels[idx]           # nodo destino
        w   = data.loc[src, dst]    # peso original
        G_tree.add_edge(src, dst, weight=w)

# Posición fija para mejor consistencia
pos = nx.spring_layout(G_tree, seed=42)

plt.figure(figsize=(6,5))
# Dibujar nodos y etiquetas
nx.draw_networkx_nodes(G_tree, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_labels(G_tree, pos, font_weight='bold')

# Dibujar solo las aristas del árbol mínimo
nx.draw_networkx_edges(
    G_tree, pos,
    edge_color='red',
    width=2,
    arrowsize=15
)

# Etiquetas de peso en esas aristas
edge_labels = nx.get_edge_attributes(G_tree, 'weight')
nx.draw_networkx_edge_labels(G_tree, pos, edge_labels=edge_labels, font_color='red')

plt.title(f"Árbol de Caminos Mínimos desde {labels[int(nodo_analizar)-1]}")
plt.axis('off')
plt.show()
