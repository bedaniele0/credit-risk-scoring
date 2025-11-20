# 🔧 SOLUCIÓN COMPLETA: Errores de Ejecución del Notebook

**Proyecto:** Credit Risk Scoring - UCI Taiwan
**Problema:** ModuleNotFoundError al ejecutar 03_model_training.ipynb
**Causa:** Notebook ejecutándose con kernel incorrecto (no usa venv)
**Autor:** Ing. Daniel Varela Pérez
**Email:** bedaniele0@gmail.com
**Fecha:** 2025-11-18

---

## 🚨 PROBLEMA IDENTIFICADO

El notebook `03_model_training.ipynb` mostró estos errores:

```
ModuleNotFoundError: No module named 'numpy'
NameError: name 'pd' is not defined
OSError: Library not loaded: libomp.dylib (LightGBM)
```

### Causa Raíz:
1. **Kernel incorrecto**: El notebook NO está usando el entorno virtual `venv/`
2. **Dependencias faltantes**: Las librerías están en `venv/` pero Jupyter usa otro Python
3. **LightGBM requiere OpenMP**: Necesita `libomp` instalado en macOS

---

## ✅ SOLUCIÓN COMPLETA (Paso a Paso)

### PASO 1: Instalar OpenMP (para LightGBM)

```bash
# Opción A: Con Homebrew (recomendado)
brew install libomp

# Si hay problemas de permisos:
sudo chown -R $(whoami) /usr/local/Homebrew
brew install libomp
```

**Verificar instalación:**
```bash
ls /usr/local/opt/libomp/lib/libomp.dylib  # Debe existir
```

---

### PASO 2: Configurar el Kernel de Jupyter con VENV

```bash
# 1. Activar entorno virtual
source venv/bin/activate

# 2. Instalar ipykernel en venv
pip install ipykernel

# 3. Registrar venv como kernel de Jupyter
python -m ipykernel install --user --name=credit-risk-venv --display-name="Python (credit-risk-venv)"

# 4. Verificar que el kernel fue creado
jupyter kernelspec list
```

**Salida esperada:**
```
Available kernels:
  credit-risk-venv    /Users/danielevarella/Library/Jupyter/kernels/credit-risk-venv
  python3             /usr/local/share/jupyter/kernels/python3
```

---

### PASO 3: Cambiar el Kernel del Notebook

#### En Jupyter Notebook:
1. Abrir `03_model_training.ipynb`
2. Ir a: **Kernel → Change kernel → Python (credit-risk-venv)**
3. Reiniciar kernel: **Kernel → Restart**

#### En VS Code:
1. Abrir `03_model_training.ipynb`
2. Click en el selector de kernel (arriba a la derecha)
3. Seleccionar: **Python (credit-risk-venv)**
4. Ejecutar primera celda para verificar

---

### PASO 4: Verificar que Todo Funciona

```bash
# Activar venv
source venv/bin/activate

# Ejecutar verificación
python -c "
import sys
print('✓ Python:', sys.executable)

import numpy, pandas, sklearn, mlflow, optuna, xgboost
print('✓ NumPy:', numpy.__version__)
print('✓ Pandas:', pandas.__version__)
print('✓ Scikit-learn:', sklearn.__version__)
print('✓ MLflow:', mlflow.__version__)
print('✓ Optuna:', optuna.__version__)
print('✓ XGBoost:', xgboost.__version__)

try:
    import lightgbm
    print('✓ LightGBM:', lightgbm.__version__)
except Exception as e:
    print('✗ LightGBM ERROR:', e)
    print('  → Solución: brew install libomp')

print('\n✅ ENTORNO LISTO PARA EJECUTAR')
"
```

---

## 🔧 SOLUCIONES ALTERNATIVAS

### Si LightGBM no funciona (problema con libomp):

#### Opción 1: Usar solo XGBoost
El notebook puede ejecutarse con XGBoost en lugar de LightGBM:
- Comentar todas las secciones de LightGBM
- Usar solo: Logistic Regression + XGBoost
- Modificar ADR-001 para usar XGBoost

#### Opción 2: Reinstalar LightGBM sin OpenMP
```bash
source venv/bin/activate
pip uninstall lightgbm
pip install lightgbm --no-binary lightgbm
```

#### Opción 3: Usar Conda (alternativa completa)
```bash
conda create -n credit-risk python=3.10
conda activate credit-risk
conda install -c conda-forge lightgbm xgboost scikit-learn pandas numpy mlflow optuna
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name=credit-risk-conda
```

---

## 📋 CHECKLIST DE VERIFICACIÓN

Antes de ejecutar el notebook, verificar:

- [ ] ✅ OpenMP instalado: `brew list libomp`
- [ ] ✅ Kernel registrado: `jupyter kernelspec list | grep credit-risk`
- [ ] ✅ Dependencias instaladas: `source venv/bin/activate && pip list | grep -E "(numpy|pandas|lightgbm|mlflow)"`
- [ ] ✅ Kernel seleccionado en notebook: **Python (credit-risk-venv)**
- [ ] ✅ Verificación exitosa: ejecutar script PASO 4

---

## 🚀 COMANDO RÁPIDO DE SETUP COMPLETO

```bash
#!/bin/bash
# Setup completo del proyecto

cd /Users/danielevarella/Desktop/credit-risk-scoring

# 1. Instalar OpenMP
echo "Instalando OpenMP..."
brew install libomp 2>/dev/null || echo "⚠️  Requiere permisos: sudo chown -R \$(whoami) /usr/local/Homebrew"

# 2. Activar venv e instalar ipykernel
echo "Configurando kernel..."
source venv/bin/activate
pip install ipykernel

# 3. Registrar kernel
python -m ipykernel install --user --name=credit-risk-venv --display-name="Python (credit-risk-venv)"

# 4. Verificar
echo ""
echo "=========================================="
echo "  VERIFICACIÓN DE INSTALACIÓN"
echo "=========================================="
python -c "import numpy, pandas, sklearn, mlflow, optuna; print('✅ ENTORNO LISTO')"

echo ""
echo "✅ SETUP COMPLETADO"
echo ""
echo "Siguiente paso:"
echo "  1. Abrir notebooks/03_model_training.ipynb"
echo "  2. Cambiar kernel a: Python (credit-risk-venv)"
echo "  3. Ejecutar notebook"
```

Guarda esto en `setup_completo.sh` y ejecuta:
```bash
chmod +x setup_completo.sh
./setup_completo.sh
```

---

## 📊 ESTADO ACTUAL DEL PROYECTO

### ✅ Completado:
- Estructura de directorios (F1)
- Problem Statement (F0)
- Diseño Arquitectónico (F2)
- EDA (F3)
- Feature Engineering (F4)
- Notebooks 01 y 02 funcionando
- **requirements.txt creado**
- **Dependencias instaladas en venv/**

### ⚠️ Pendiente:
- **Configurar kernel correcto** ← ESTO ES LO CRÍTICO
- Instalar OpenMP (para LightGBM)
- Ejecutar notebook 03_model_training.ipynb
- Validación del modelo (F6)

---

## 🆘 SI AÚN HAY PROBLEMAS

### Error: "kernel not found"
```bash
# Listar kernels disponibles
jupyter kernelspec list

# Eliminar kernel antiguo (si existe)
jupyter kernelspec uninstall credit-risk-venv

# Reinstalar
source venv/bin/activate
python -m ipykernel install --user --name=credit-risk-venv --display-name="Python (credit-risk-venv)"
```

### Error: "libomp.dylib not found" (macOS)
```bash
# Verificar si libomp está instalado
brew list libomp

# Si no está:
brew install libomp

# Verificar ubicación
ls -l /usr/local/opt/libomp/lib/libomp.dylib
```

### Error: "Permission denied" en Homebrew
```bash
# Cambiar ownership
sudo chown -R $(whoami) /usr/local/Homebrew /usr/local/bin /usr/local/etc /usr/local/lib /usr/local/share /usr/local/var

# Reintentar
brew install libomp
```

---

## 📞 CONTACTO

**Ing. Daniel Varela Pérez**
**Email:** bedaniele0@gmail.com
**Tel:** +52 55 4189 3428
**Metodología:** DVP-PRO v2.0

---

**Última actualización:** 2025-11-18
**Versión:** 1.0
