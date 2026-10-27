# SAVI - Parte 6: Otimização Aplicada a Visão por Computador (Fotometria e Alinhamento Geométrico)

Este documento apresenta a fundamentação teórica, formulações matemáticas, assinaturas de API do OpenCV/SciPy/NumPy e blocos de código isolados necessários para resolver os problemas da **Parte 6**.

---

## 1. Exercício 1: Correção Fotométrica por Brilho Médio e Contraste Alvo

### Fundamentação Teórica & Matemática

Uma transformação radiométrica / afim de intensidade numa imagem $I(x, y)$ é modelada por um ganho (escala/contraste) $s \in \mathbb{R}$ e um deslocamento (_bias_/brilho) $b \in \mathbb{R}$:

$$I'(x, y) = s \cdot I(x, y) + b$$

Para uma imagem com $N$ píxeis, as métricas estatísticas de interesse são:
- **Brilho médio ($\mu$):**
  $$\mu(I') = \frac{1}{N} \sum_{x, y} I'(x, y)$$
- **Contraste ($\sigma$, desvio padrão):**
  $$\sigma(I') = \sqrt{\frac{1}{N} \sum_{x, y} \left( I'(x, y) - \mu(I') \right)^2}$$

Dado um brilho alvo $\mu_{\text{target}}$ e um contraste alvo $\sigma_{\text{target}}$, o vetor de resíduos $\mathbf{r}(s, b)$ para otimização por mínimos quadrados pode ser formulado como:

$$\mathbf{r}(s, b) = \begin{bmatrix} \mu(I') - \mu_{\text{target}} \\ \sigma(I') - \sigma_{\text{target}} \end{bmatrix}$$

O estimador de mínimos quadrados não-lineares minimiza a função de custo escalar:
$$f(s, b) = \frac{1}{2} \|\mathbf{r}(s, b)\|_2^2 = \frac{1}{2} \left[ (\mu(I') - \mu_{\text{target}})^2 + (\sigma(I') - \sigma_{\text{target}})^2 \right]$$

> **Nota Teórica:** Se apenas o erro de brilho for considerado ($\mathbf{r} = [\mu(I') - \mu_{\text{target}}]$), o sistema tem 1 equação e 2 incógnitas ($s, b$), resultando em infinitas soluções degeneradas. A adição do termo de desvio padrão isola uma solução única.

### Mecânica de API & Assinaturas

- `scipy.optimize.least_squares(fun, x0, args=(), bounds=(-inf, inf), method='trf', ...)`
  - `fun`: Função objetivo `f(params, *args) -> np.ndarray` que devolve um vetor 1D com os resíduos.
  - `x0`: Estimativa inicial dos parâmetros (array 1D).
  - Devolve: Objeto `OptimizeResult` com atributo `x` contendo os parâmetros ótimos estimados.
- Operações de ponto flutuante em NumPy:
  - `np.mean(a, axis=None, dtype=np.float64)`
  - `np.std(a, axis=None, dtype=np.float64)`
  - `np.clip(a, a_min, a_max)`: Garante a saturação física dos canais de cor $[0, 255]$ após aplicar o modelo afim.

### Exemplo Mínimo Isolado

```python
import numpy as np
from scipy.optimize import least_squares

def residual_brightness_contrast(params: np.ndarray, img_gray: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    s, b = params[0], params[1]
    # Aplicar o modelo afim
    transformed = s * img_gray + b
    
    # Calcular métricas estatísticas
    current_mean = np.mean(transformed)
    current_std = np.std(transformed)
    
    # Vetor de resíduos devolvido ao otimizador
    return np.array([current_mean - target_mean, current_std - target_std], dtype=np.float64)

# Exemplo sintético de invocação:
dummy_img = np.random.uniform(50, 180, size=(100, 100)).astype(np.float64)
x0 = np.array([1.0, 0.0])  # s=1, b=0
res = least_squares(residual_brightness_contrast, x0, args=(dummy_img, 80.0, 60.0))
s_opt, b_opt = res.x
```

---

## 2. Exercício 2 & 3: Correção de Cor num Mosaico (Alinhamento Fotométrico)

### Fundamentação Teórica & Matemática

Dadas duas imagens sobrepostas, a imagem de treino/referência $I_t(x, y)$ e a imagem query transformada/alinhada espacialmente $I_q(x, y)$:
- A máscara binária da região de sobreposição $\Omega$ é definida por:
  $$\Omega = \{ (x, y) \mid I_t(x, y) \text{ é válido} \land I_q(x, y) \text{ é válido} \}$$
  (Por exemplo, onde os píxeis de $I_q$ não são nulos após a transformação).

#### Exercício 2 (Similaridade Pixel a Pixel):
O objetivo é minimizar a diferença fotométrica ponto a ponto na máscara de sobreposição $\Omega$:
$$r_{xy}(s, b) = (s \cdot I_q(x, y) + b) - I_t(x, y), \quad \forall (x, y) \in \Omega$$

- **Problema de Degenerescência:** Se otimizarmos simultaneamente os parâmetros de $I_t$ ($s_t, b_t$) e de $I_q$ ($s_q, b_q$) sem referência externa, a solução ótima trivial é $s_t = s_q = 0$ e $b_t = b_q = 0$, tornando ambas as imagens pretas com erro zero ($0 - 0 = 0$). Fixar $I_t$ impede a degenerescência.
- **Resíduo escalar vs. Vectorial:** Devolver a média escalar dos erros $\frac{1}{|\Omega|} \sum r_{xy}$ permite que erros positivos e negativos se anulem mutuamente; devolver o vetor de resíduos completo permite aos métodos de Gauss-Newton / Levenberg-Marquardt calcular a matriz Jacobiana completa e convergir corretamente.

#### Exercício 3 (Equilíbrio de Pesos Multi-Objetivo):
Ao combinar resíduos pixel-a-pixel com restrições globais de brilho e contraste, surge uma disparidade dimensional: existem $M = |\Omega| \gg 10^5$ resíduos de similaridade e apenas 4 resíduos estatísticos globais. Como o estimador minimiza $\sum_i r_i^2$, termos estatísticos não ponderados tornam-se negligenciáveis.

Para equilibrar a contribuição, introduz-se um fator de escala/ponderação $w$:
$$w = \sqrt{\frac{|\Omega|}{K}}$$
onde $K$ é o número de termos estatísticos (ex: 2 ou 4), normalizando a variância das duas funções de perda.

### Mecânica de API & Assinaturas

- `cv.inRange(src, lowerb, upperb)` ou indexação booleana NumPy para identificação de máscara:
  `mask = (img_warped > 0)`
- Indexação booleana para extração de arrays 1D unidimensionais:
  `vals_q = img_warped[mask]`
  `vals_t = img_train[mask]`

### Exemplo Mínimo Isolado

```python
import numpy as np
from scipy.optimize import least_squares

def residual_similarity(params: np.ndarray, q_vals: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    s, b = params[0], params[1]
    # Diferença pixel a pixel nos pontos válidos da sobreposição
    return (s * q_vals + b) - t_vals

# Array 1D de píxeis na região de sobreposição
q_overlap = np.array([50, 100, 150], dtype=np.float64)
t_overlap = np.array([70, 120, 170], dtype=np.float64)

x0 = np.array([1.0, 0.0])
res = least_squares(residual_similarity, x0, args=(q_overlap, t_overlap))
```

---

## 3. Exercício 4: Alinhamento Geométrico por Features SIFT e Otimização 2D

### Fundamentação Teórica & Matemática

Se a relação geométrica entre $I_q$ e $I_t$ for uma translação pura parametrizada pelo vetor $\mathbf{t} = [\Delta x, \Delta y]^T$:
$$\begin{bmatrix} x_t \\ y_t \end{bmatrix} = \begin{bmatrix} x_q \\ y_q \end{bmatrix} + \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix}$$

Dado um conjunto de $P$ correspondências de pontos chave $\{(\mathbf{p}_{q,i}, \mathbf{p}_{t,i})\}_{i=1}^P$:
O erro residual de reprojeção para o $i$-ésimo par de pontos é dado por:
$$\mathbf{r}_i(\Delta x, \Delta y) = \begin{bmatrix} (x_{q,i} + \Delta x) - x_{t,i} \\ (y_{q,i} + \Delta y) - y_{t,i} \end{bmatrix}$$

O custo total de mínimos quadrados é:
$$E(\Delta x, \Delta y) = \sum_{i=1}^P \left[ (x_{q,i} + \Delta x - x_{t,i})^2 + (y_{q,i} + \Delta y - y_{t,i})^2 \right]$$

Derivando em ordem a $\Delta x$:
$$\frac{\partial E}{\partial \Delta x} = 2 \sum_{i=1}^P (x_{q,i} + \Delta x - x_{t,i}) = 0 \implies \Delta x^* = \frac{1}{P} \sum_{i=1}^P (x_{t,i} - x_{q,i})$$
Isto demonstra analiticamente por que o estimador de mínimos quadrados coincide exatamente com a média das diferenças ponto a ponto.

### Mecânica de API & Assinaturas

- `cv.SIFT_create()`
  - `sift.detectAndCompute(image, mask=None) -> (keypoints, descriptors)`
    - `keypoints`: Lista de objetos `cv.KeyPoint`, onde cada elemento possui `.pt = (x, y)`.
    - `descriptors`: Matriz `np.ndarray` do tipo `float32` de dimensões `(N_pts, 128)`.
- `cv.BFMatcher(normType=cv.NORM_L2, crossCheck=False)`
  - `matcher.knnMatch(queryDescriptors, trainDescriptors, k=2) -> list[list[cv.DMatch]]`
- `cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, outImg, ...)`

### Exemplo Mínimo Isolado

```python
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

# 1. Extração SIFT e Emparelhamento
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_L2)
raw_matches = bf.knnMatch(des1, des2, k=2)

# Teste de rácio de Lowe (Lowe's ratio test)
good_pts1 = []
good_pts2 = []
for m, n in raw_matches:
    if m.distance < 0.75 * n.distance:
        good_pts1.append(kp1[m.queryIdx].pt)
        good_pts2.append(kp2[m.trainIdx].pt)

pts_q = np.float32(good_pts1) # Shape: (N, 2)
pts_t = np.float32(good_pts2) # Shape: (N, 2)

# 2. Resíduos de Translação 2D
def residual_translation(delta: np.ndarray, p_query: np.ndarray, p_train: np.ndarray) -> np.ndarray:
    # delta = [dx, dy]
    p_trans = p_query + delta
    diff = p_trans - p_train
    return diff.flatten() # Retorna vetor 1D de tamanho 2*N

res = least_squares(residual_translation, x0=np.array([0.0, 0.0]), args=(pts_q, pts_t), bounds=([-500, -500], [500, 500]))
dx_opt, dy_opt = res.x
```

---

## 4. Exercício 5: Rejeição de Outliers via RANSAC e Funções de Perda Robustas

### Fundamentação Teórica & Matemática

O método clássico dos mínimos quadrados ordinários assume que os resíduos seguem uma distribuição Gaussiana. A presença de correspondências incorretas (_outliers_) causa distorções severas na estimativa porque o erro quadrático penaliza desvios com $\mathcal{O}(r^2)$.

Para mitigar o impacto de outliers:
1. **RANSAC (Random Sample Consensus):** Algoritmo iterativo que seleciona subconjuntos mínimos de correspondências para gerar hipóteses de modelos e conta o número de inliers cuja distância de reprojeção é inferior a um limiar $\tau$.
2. **Estimadores M robustos (Loss Functions no `least_squares`):**
   - **Soft L1 (`loss='soft_l1'`):** Transforma a função de perda quadrática clássica $\rho(z) = z$ em:
     $$\rho(z) = 2(\sqrt{1 + z} - 1), \quad \text{onde } z = r^2$$
     Para valores pequenos de resíduo ($z \to 0$), comporta-se como $L_2$ ($\rho(z) \approx z$). Para resíduos grandes ($z \to \infty$), o declive atenua-se para $\frac{d\rho}{dz} \approx \frac{1}{\sqrt{z}}$, tornando o custo sublinear e imune a discrepâncias extremas.

### Mecânica de API & Assinaturas

- `cv.findHomography(srcPoints, dstPoints, method, ransacReprojThreshold=...)`
  - `method`: `cv.RANSAC` ou `cv.USAC_MAGSAC`.
  - `ransacReprojThreshold`: Limiar de erro de reprojeção máximo em píxeis para tratar um ponto como inlier.
  - Devolve: `(H, mask)` onde `mask` é um array binário indicando inliers (1) e outliers (0).
- Argumento `loss` no `least_squares`:
  - `least_squares(..., loss='soft_l1', f_scale=1.0)`: `f_scale` define o ponto de transição entre a região linear e a região atenuada.

### Exemplo Mínimo Isolado

```python
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

# Deteção e filtragem geométrica via RANSAC
# pts_q, pts_t com shape (N, 1, 2) ou (N, 2)
H, inlier_mask = cv.findHomography(pts_q, pts_t, cv.RANSAC, ransacReprojThreshold=5.0)

# Otimização robusta com atenuação de perdas
res_robust = least_squares(
    residual_translation,
    x0=np.array([0.0, 0.0]),
    args=(pts_q, pts_t),
    loss='soft_l1',
    f_scale=2.0
)
```

---

## 5. Exercício 6: Alinhamento Geométrico e Fotométrico Direto (Pixel-Based)

### Fundamentação Teórica & Matemática

Nesta abordagem direta (sem extração prévia de features esparsas), minimiza-se diretamente a discrepância de intensidade em cada pixel sob uma translação espacial contínua $(\Delta x, \Delta y) \in \mathbb{R}^2$ e uma escala/bias fotométricos $(s, b) \in \mathbb{R}^2$:

$$e(x, y) = s \cdot I_q(x, y) + b - I_t(x + \Delta x, y + \Delta y), \quad \forall (x, y) \in \text{dom}(I_q)$$

O vetor de parâmetros a estimar em conjunto é:
$$\mathbf{\theta} = [\Delta x, \Delta y, s, b]^T$$

#### Interpolação Sub-pixel:
Como $(\Delta x, \Delta y)$ assume valores contínuos em $\mathbb{R}$, as coordenadas amostradas na imagem de treino $(x + \Delta x, y + \Delta y)$ não caem exatamente no reticulado discreto de píxeis inteiros. É indispensável recorrer à interpolação bilinear via remapeamento de coordenadas.

#### Paisagem Não-Convexa e Armadilhas de Mínimos Locais:
A função de erro baseada em intensidades pixel-a-pixel é fortemente não-convexa, apresentando inúmeros mínimos locais causados por texturas de alta frequência. Para assegurar convergência:
1. **Pirâmide Gaussiana / Multiescala:** Redução da escala (subamostragem) e aplicação de suavização (`cv.GaussianBlur`), eliminando altas frequências espaciais e alargando a bacia de atração do mínimo global.
2. **Estimativa Inicial:** Inicializar com Template Matching (`cv.matchTemplate`), que fornece uma aproximação global de $\Delta x_0, \Delta y_0$ ao nível de número inteiro.

### Mecânica de API & Assinaturas

- `cv.remap(src, map1, map2, interpolation, borderMode=..., borderValue=...)`
  - `map1`: Matriz float32 com as coordenadas $x$ de amostragem.
  - `map2`: Matriz float32 com as coordenadas $y$ de amostragem.
  - `interpolation`: `cv.INTER_LINEAR` para interpolação bilinear.
- `cv.matchTemplate(image, templ, method)`
  - `method`: `cv.TM_CCOEFF_NORMED` (valores entre -1 e 1; pico em 1).
- `cv.minMaxLoc(src)`
  - Devolve: `(min_val, max_val, min_loc, max_loc)`.

### Exemplo Mínimo Isolado

```python
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

def sample_with_translation(img: np.ndarray, dx: float, dy: float, out_shape: tuple) -> np.ndarray:
    h, w = out_shape[:2]
    # Geração do reticulado de coordenadas x, y deslocadas
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + np.float32(dx)
    map_y = grid_y + np.float32(dy)
    # Remapeamento bilinear
    warped = cv.remap(img, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    return warped

def residual_joint(params: np.ndarray, img_q: np.ndarray, img_t: np.ndarray) -> np.ndarray:
    dx, dy, s, b = params
    # Obter imagem de treino amostrada nas coordenadas da query deslocada
    t_sampled = sample_with_translation(img_t, dx, dy, img_q.shape)
    
    # Modelo afim aplicado à query
    q_transformed = s * img_q + b
    
    # Máscara de validade (ignorar bordas fora dos limites)
    valid_mask = (t_sampled > 0)
    
    diff = q_transformed[valid_mask] - t_sampled[valid_mask]
    return diff.flatten()

# Template Matching para Inicialização Global:
res_tm = cv.matchTemplate(img_train_small, img_query_small, cv.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv.minMaxLoc(res_tm)
init_dx, init_dy = max_loc

# Otimização Fina Contínua
x0 = np.array([init_dx, init_dy, 1.0, 0.0], dtype=np.float64)
opt_res = least_squares(residual_joint, x0, args=(img_query_small, img_train_small))
```
