# SAVI - Parte 5: Otimização e Ajuste de Modelos

## Visão Geral

Esta unidade aborda técnicas de recolha de dados interativa e algoritmos de otimização numérica para ajuste de curvas (model fitting):
1. **Recolha interativa de dados (GUI Event Handling com Matplotlib)** e persistência em JSON.
2. **Ajuste Estocástico / Aleatório (Random Line Fitting)** via amostragem de Monte Carlo e avaliação de métricas de erro.
3. **Ajuste Numérico por Mínimos Quadrados (Nonlinear Least Squares)** utilizando o algoritmo Levenberg-Marquardt / Trust Region Reflective via `scipy.optimize.least_squares`.
4. **Modelos Não-Lineares / Por Ramos (Composed/Piecewise Model Fitting)** combinando diferentes regimes funcionais ($x \le 0$ e $x > 0$).

---

## 1. Exercício 1: Geração Interativa de Pontos (`generate_data`)

### Fundamento Teórico & Event Loop
A interface gráfica do Matplotlib opera através de um loop de eventos acionado por interrupções do sistema operativo (ratos e teclados).
Para registar amostras discretas do utilizador:
- Liga-se o evento de clique `button_press_event` a uma função de callback.
- Converte-se a posição física do cursor em coordenadas normalizadas do sistema de eixos: $(x_{\text{data}}, y_{\text{data}})$.
- Interrompe-se a recolha através do evento de teclado `key_press_event` ao intercetar a tecla `escape`.

### Mecânica de Implementação e Assinaturas
- Registo de listeners: `fig.canvas.mpl_connect(event_name: str, callback_fn: Callable)`.
- Exportação estruturada para dicionário: `{"xs": [...], "ys": [...]}`.
- Serialização via `json.dump(obj, fp, indent=4)`.

```python
import matplotlib.pyplot as plt
import json

xs, ys = [], []

def on_click(event):
    if event.inaxes is not None:
        xs.append(float(event.xdata))
        ys.append(float(event.ydata))
        plt.plot(event.xdata, event.ydata, 'ro')
        event.canvas.draw_idle()

def on_key(event):
    if event.key == 'escape':
        plt.close()

fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-5, 5)
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

with open("points.json", "w") as f:
    json.dump({'xs': xs, 'ys': ys}, f, indent=4)
```

---

## 2. Exercício 2: Random Line Fitting (Monte Carlo Search)

### Formulação Matemática
Dado um conjunto de $N$ pontos $\{(x_i, y'_i)\}_{i=0}^{N-1}$ e o modelo afim:

$$y_i = m \cdot x_i + b$$

O objetivo é encontrar os parâmetros $\mathbf{p}^* = (m^*, b^*)$ que minimizam o erro acumulado. Para evitar o cancelamento entre erros positivos e negativos, calcula-se a norma $L_1$ (MAE) ou $L_2$ (MSE):

$$J_{L1}(m, b) = \sum_{i=0}^{N-1} |(m \cdot x_i + b) - y'_i|$$

$$J_{L2}(m, b) = \sum_{i=0}^{N-1} ((m \cdot x_i + b) - y'_i)^2$$

### Mecânica Numérica
Numa busca aleatória uniforme dentro de limites pré-definidos $[m_{\min}, m_{\max}]$ e $[b_{\min}, b_{\max}]$:
- Sorteiam-se candidatos: `m = np.random.uniform(m_low, m_high)`.
- Avalia-se o custo vetorial sobre NumPy: `err = np.sum(np.abs((m * x + b) - y))`.
- Mantém-se o par com menor custo acumulado.

```python
import numpy as np

def evaluate_random_line(x_arr, y_arr, n_iter=1000):
    best_error = float('inf')
    best_params = (0.0, 0.0)
    
    for _ in range(n_iter):
        m_cand = np.random.uniform(-3.0, 3.0)
        b_cand = np.random.uniform(-5.0, 5.0)
        
        y_pred = m_cand * x_arr + b_cand
        error = np.sum(np.abs(y_pred - y_arr))
        
        if error < best_error:
            best_error = error
            best_params = (m_cand, b_cand)
            
    return best_params, best_error
```

---

## 3. Exercício 3: Ajuste por Mínimos Quadrados (`scipy.optimize.least_squares`)

### Formulação do Otimizador
Diferente da busca estocástica cega, os métodos de gradiente convergem deterministicamente para mínimos locais/globais.
O problema é expresso na forma vetorial:

$$\min_{\mathbf{p}} \frac{1}{2} \|\mathbf{r}(\mathbf{p})\|_2^2 = \min_{\mathbf{p}} \frac{1}{2} \sum_{i=0}^{N-1} r_i(\mathbf{p})^2$$

Onde $\mathbf{r}(\mathbf{p}) \in \mathbb{R}^N$ é o **vetor de resíduos**:

$$r_i(\mathbf{p}) = y_{\text{model}}(x_i; \mathbf{p}) - y'_i$$

### Mecânica da API SciPy
- **Assinatura:** `scipy.optimize.least_squares(fun, x0, args=(), ...)`
- `fun`: Função do tipo `fun(params, *args) -> np.ndarray`. Deve devolver um **vetor 1D de resíduos com dimensionalidade $N$**, e **não** uma soma escalar.
- `x0`: Array de estimativa inicial dos parâmetros.
- O resultado é acedido no atributo `.x` do objeto retornado.

```python
from scipy.optimize import least_squares
import numpy as np

def residuals(params, x, y):
    m, b = params
    return (m * x + b) - y

# Execução do otimizador com x0 e argumentos adicionais
initial_guess = [1.0, 0.0]
res = least_squares(residuals, x0=initial_guess, args=(x_data, y_data))
m_opt, b_opt = res.x
```

---

## 4. Exercício 4: Composed Model Fitting (Modelo por Ramos)

### Definição Matemática
O modelo divide o domínio de $x$ em duas regiões funcionais distintas:

$$y(x; \mathbf{p}) = \begin{cases} 
m \cdot x + b_1 & \text{se } x \le 0 \\
a \cdot x^2 + c \cdot x + b_2 & \text{se } x > 0 
\end{cases}$$

Onde $\mathbf{p} = [m, b_1, a, c, b_2] \in \mathbb{R}^5$.

### Mecânica de Vetorização (`np.where`)
A avaliação em NumPy de múltiplos ramos é efetuada sem loops explícitos através de máscaras booleanas:

```python
def predict_composed(x, m, b1, a, c, b2):
    return np.where(x <= 0, m * x + b1, a * (x ** 2) + c * x + b2)
```

O resíduo continua a ser a diferença elemento a elemento:

```python
def composed_residuals(params, x, y):
    y_pred = predict_composed(x, *params)
    return y_pred - y
```
