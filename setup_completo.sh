#!/bin/bash
# Setup completo del proyecto Credit Risk Scoring
# Autor: Ing. Daniel Varela Pérez
# Email: bedaniele0@gmail.com
# Fecha: 2025-11-18
# Metodología: DVP-PRO v2.0

echo "================================================"
echo "  SETUP COMPLETO: Credit Risk Scoring"
echo "  Metodología DVP-PRO v2.0"
echo "================================================"
echo ""

cd /Users/danielevarella/Desktop/credit-risk-scoring

# 1. Instalar OpenMP
echo "PASO 1: Instalando OpenMP (necesario para LightGBM)..."
echo "------------------------------------------------------"
if ! brew list libomp &>/dev/null; then
    echo "Instalando libomp con Homebrew..."
    if brew install libomp 2>/dev/null; then
        echo "✅ OpenMP instalado correctamente"
    else
        echo "⚠️  Error al instalar OpenMP. Posible solución:"
        echo "    sudo chown -R \$(whoami) /usr/local/Homebrew"
        echo "    brew install libomp"
        echo ""
        echo "ALTERNATIVA: El notebook puede ejecutarse sin LightGBM usando solo XGBoost"
    fi
else
    echo "✅ OpenMP ya está instalado"
fi
echo ""

# 2. Verificar y activar venv
echo "PASO 2: Configurando entorno virtual..."
echo "------------------------------------------------------"
if [ ! -d "venv" ]; then
    echo "❌ ERROR: No existe el directorio venv/"
    echo "   Crear con: python3 -m venv venv"
    exit 1
fi

source venv/bin/activate
echo "✅ Entorno virtual activado"
echo ""

# 3. Instalar ipykernel
echo "PASO 3: Instalando ipykernel..."
echo "------------------------------------------------------"
pip install ipykernel --quiet
echo "✅ ipykernel instalado"
echo ""

# 4. Registrar kernel de Jupyter
echo "PASO 4: Registrando kernel de Jupyter..."
echo "------------------------------------------------------"
python -m ipykernel install --user --name=credit-risk-venv --display-name="Python (credit-risk-venv)"
echo "✅ Kernel registrado: credit-risk-venv"
echo ""

# 5. Verificar instalación de dependencias críticas
echo "PASO 5: Verificando dependencias..."
echo "------------------------------------------------------"
python -c "
import sys
import importlib

print('Python:', sys.executable)
print('')

packages = {
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'Scikit-learn',
    'mlflow': 'MLflow',
    'optuna': 'Optuna',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM'
}

errors = []
for pkg, name in packages.items():
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'✅ {name}: {version}')
    except Exception as e:
        print(f'❌ {name}: ERROR - {str(e)}')
        errors.append(name)

print('')
if errors:
    print(f'⚠️  Paquetes con error: {len(errors)}')
    for pkg in errors:
        print(f'   - {pkg}')
    print('')
    print('Solución: pip install -r requirements.txt')
else:
    print('✅ TODAS LAS DEPENDENCIAS INSTALADAS CORRECTAMENTE')
"

echo ""
echo "================================================"
echo "  ✅ SETUP COMPLETADO"
echo "================================================"
echo ""
echo "Kernels de Jupyter disponibles:"
jupyter kernelspec list | grep -E "(Available|credit-risk)"
echo ""
echo "================================================"
echo "  SIGUIENTE PASO:"
echo "================================================"
echo "  1. Abrir: notebooks/03_model_training.ipynb"
echo "  2. Cambiar kernel a: Python (credit-risk-venv)"
echo "     - En Jupyter: Kernel → Change kernel → Python (credit-risk-venv)"
echo "     - En VS Code: Click en selector de kernel (arriba derecha)"
echo "  3. Ejecutar notebook completo"
echo ""
echo "================================================"
echo "  DOCUMENTACIÓN:"
echo "================================================"
echo "  - Guía completa: SOLUCION_ERRORES.md"
echo "  - Fix LightGBM: fix_lightgbm.sh"
echo "  - Dependencias: requirements.txt"
echo ""
echo "Contacto: bedaniele0@gmail.com"
echo "================================================"
