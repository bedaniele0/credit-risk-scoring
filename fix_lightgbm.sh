#!/bin/bash
# Fix LightGBM installation - Install OpenMP
# Autor: Ing. Daniel Varela Pérez
# Email: bedaniele0@gmail.com

echo "================================================"
echo "  SOLUCIÓN: Instalación de LightGBM con OpenMP"
echo "================================================"

echo ""
echo "OPCIÓN 1: Instalar libomp con Homebrew (recomendado)"
echo "Ejecuta manualmente:"
echo "  brew install libomp"
echo ""
echo "Si tienes problemas de permisos:"
echo "  sudo chown -R \$(whoami) /usr/local/Homebrew"
echo "  brew install libomp"
echo ""

echo "OPCIÓN 2: Usar Conda (alternativa)"
echo "  conda install -c conda-forge lightgbm"
echo ""

echo "OPCIÓN 3: Compilar desde fuente"
echo "  pip uninstall lightgbm"
echo "  pip install lightgbm --install-option=--nomp"
echo ""

echo "================================================"
echo "  SOLUCIÓN RÁPIDA (sin LightGBM temporalmente)"
echo "================================================"
echo ""
echo "El notebook puede ejecutarse con XGBoost en lugar de LightGBM."
echo "Para continuar SIN LightGBM:"
echo "  1. Ejecuta el notebook con solo XGBoost"
echo "  2. Instala libomp más tarde con: brew install libomp"
echo ""
