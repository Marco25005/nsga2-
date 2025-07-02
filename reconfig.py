# -*- coding: utf-8 -*-
"""
Módulo que contiene la función principal de reconfiguración de red.

Traducido de la función MATLAB reconfig.m.
Carga datos de red, evalúa el caso inicial y presenta un menú
para ejecutar diferentes algoritmos de optimización (GA, NSGA-II, SLE, etc.).
"""

import time # Para tic/toc
from tnetwork import TNetwork

def reconfig(filename):
    """
    Función principal para la reconfiguración de red.

    Carga datos de red desde un archivo, evalúa el caso inicial y
    presenta un menú interactivo para ejecutar diferentes algoritmos
    de optimización.

    Args:
        filename (str): Nombre del archivo (ej: Excel) que contiene los datos de la red.
    """
    print("Iniciando herramienta de reconfiguración...")

    # Crear una instancia de la red usando la clase TNetwork
    net = TNetwork()

    # Cargar datos de la red
   
    net.getdata(filename)
       
    # Configurar propiedades de la red
    net.isimprove = True # Habilitar la mejora local después de la evaluación
    net.time = 24        # Número de pasos de tiempo (ej: horas en un día)

    # Evaluar el caso inicial (estado por defecto de las ramas en los datos)
    print("\nEvaluando caso inicial...")
    start_time = time.time() # tic en MATLAB
   
    net.evaluate_on(None) # evaluate_on espera None o una lista de ison

    end_time = time.time() # toc en MATLAB
    elapsed_time = end_time - start_time
    print(f"Evaluación inicial completada en {elapsed_time:.4f} segundos.")

    # Verificar si la evaluación inicial fue exitosa
    if net.status['error'] != 0:
        # error('No hay solucion para el caso inicial') en MATLAB
        print("\nERROR: No hay solución válida para el caso inicial.")
        print(f"Código de error: {net.status['error']}")
        # Opcional: mostrar el estado de error detallado
        # net.printstatus()
        # En Python, es mejor lanzar una excepción o simplemente salir
        # raise RuntimeError("No hay solución para el caso inicial")
        return # Salir de la función

    # Mostrar el estado inicial
    print('\n----- Caso Inicial --------')
    net.printstatus() # printstatus debe estar implementado en TNetwork
    # Bucle principal del menú
    

    start_time = time.time() # tic en MATLAB
        # Ejecutar la opción seleccionada
    npop = 20
                # Asumimos que nsga2solver1 es un método de la clase TNetwork
                # El segundo argumento 'pop' parece ser la población inicial (puede ser None/[] si se genera aleatoriamente)
    net.nsga2solver1(npop, []) # Pasar lista vacía para la población inicial
       

# Ejemplo de cómo ejecutar la herramienta (si este módulo es el script principal)
if __name__ == "__main__":
    # Puedes cambiar 'datos.xlsx' al nombre de tu archivo de datos
    # Asegúrate de que 'datos.xlsx' y 'exgraphs.xlsx' estén en el mismo directorio
    # o proporciona la ruta completa.
    data_file = 'ex136barras.xlsx'
    reconfig(data_file)

   
    
