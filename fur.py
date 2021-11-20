from taichi import ti
from perlin_noise import PerlinNoise

PI = 3.141592653589793

ti.init(arch=ti.gpu, debug=True)

uvScale = 1.0
colorUvScale = 0.1
furLayers = 64
furDepth = 0.2
rayStep = furDepth * 2.0 / float(furLayers)
furThreshold = 0.40
shininess = 50.0

iTime = ti.field(ti.f32, ())

n = 320
iResolution = ti.Vector([n * 2, n])

pixels = ti.Vector.field(3, ti.f32, shape=(n * 2, n))
noiseAlpha = PerlinNoise(90, 1.0)
noiseDepth = PerlinNoise(120, 1.0)
noiseColor = PerlinNoise(120, 1.5)

@ti.func
def length(v):
    l = ti.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    return l

@ti.func
def normalize(v):
    l = length(v)
    v /= l
    return v

@ti.func
def clamp(t, x, y):
    return ti.min(ti.max(t, x), y)

@ti.func
def smoothstep(edge0, edge1, x):
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

@ti.func
def atan2(y, x):
    r = 0.0
    if ti.abs(x) < 0.00001:
        r = PI / 2.0 if y >= 0.0 else -PI / 2.0
    else:
        z = ti.sqrt(x * x + y * y)
        r = ti.asin(y / z)
        if x < 0.0:
            r += PI
            if r > PI:
                r -= 2.0 * PI
            r = -r
    return r

@ti.func
def rotateX(p, a):
    sa = ti.sin(a)
    ca = ti.cos(a)
    return ti.Vector([p.x, ca * p.y - sa * p.z, sa * p.y + ca * p.z])

@ti.func
def rotateY(p, a):
    sa = ti.sin(a)
    ca = ti.cos(a)
    return ti.Vector([ca * p.x + sa * p.z, p.y, -sa * p.x + ca * p.z])

@ti.func
def intersectSphere(ro, rd, r):
    b = (-ro).dot(rd)
    det = b * b - ro.dot(ro) + r * r
    t = -1.0
    if(det >= 0.0):
        det = ti.sqrt(det)
        t = b - det
    return t

@ti.func
def cartesianToSpherical(p):
    r = length(p)
    t = (r - (1.0 - furDepth)) / furDepth
    p = rotateX(ti.Vector([p.z, p.y, p.x]), -ti.cos(iTime[None] * 1.5) * t * t * 0.4)
    q = ti.Vector([p.z, p.y, p.x])

    q /= r
    uv = ti.Vector([atan2(q.y, q.x), ti.acos(q.z)])
    uv.y -= t * t * 0.1
    return uv

@ti.func
def furDensity(pos):
    uv = cartesianToSpherical(ti.Vector([pos.x, pos.z, pos.y]))
    sample = ti.Vector([(uv.x + PI) / PI, (uv.y + PI / 2.0) / PI])
    alpha = noiseAlpha.eval(sample.x * uvScale, sample.y * uvScale, 0.0)

    density = smoothstep(furThreshold, 0.6, alpha)
    r = length(pos)
    t = (r - (1.0 - furDepth)) / furDepth

    len = noiseDepth.eval(sample.x * uvScale, sample.y * uvScale, 0.0)
    density *= smoothstep(len, len - 0.2, t)

    return density, uv

@ti.func
def furNormal(pos, density):
    eps = 0.01
    x, _ = furDensity(ti.Vector([pos.x + eps, pos.y, pos.z]))
    y, _ = furDensity(ti.Vector([pos.x, pos.y + eps, pos.z]))
    z, _ = furDensity(ti.Vector([pos.x, pos.y, pos.z + eps]))
    return normalize(ti.Vector([x, y, z]) - ti.Vector([density, density, density]))

@ti.func
def furShade(pos, uv, ro, density):
    L = ti.Vector([0.0, 1.0, 0.0])
    V = normalize(ro - pos)
    H = normalize(V + L)

    N = -furNormal(pos, density)
    # N = normalize(pos)
    diff = ti.max(0.0, N.dot(L) * 0.6 + 0.6)
    spec = ti.pow(max(0.0, N.dot(H)), shininess)

    sample = ti.Vector([(uv.x + PI) / PI, (uv.y + PI / 2.0) / PI])
    color = noiseColor.evalColor(sample.x * colorUvScale, sample.y * colorUvScale, 0.0)
    r = length(pos)
    t = (r - (1.0 - furDepth)) / furDepth
    t = clamp(t, 0.0, 1.0)
    i = t * 0.5 + 0.5
    spec *= i
    return color * diff * i + ti.Vector([spec, spec, spec])

@ti.func
def scene(ro, rd):
    p = ti.Vector([0.0, 0.0, 0.0])
    r = 1.0
    t = intersectSphere(ro - p, rd, r)
    c = ti.Vector([0.0, 0.0, 0.0, 0.0])
    if t > 0.0:
        pos = ro + rd * t
        # color = furShade(pos, ti.Vector([0.0, 0.0]), ro, 0.0)
        # c = ti.Vector([color.x, color.y, color.z, 1.0])
        for i in range(furLayers):
            alpha, uv = furDensity(pos)
            if alpha > 0.0:
                color = furShade(pos, uv, ro, alpha)
                sampleCol = ti.Vector([color.x * alpha, color.y * alpha, color.z * alpha, alpha])
                c = c + sampleCol * (1.0 - c.w)
                if c.w > 0.95:
                    break
            pos += rd * rayStep
    return c

@ti.kernel
def paint():
    roty = ti.sin(iTime[None] * 1.5)
    for i, j in pixels:
        uv = 1.0 * ti.Vector([i, j]) / iResolution
        uv = uv * 2.0 - 1.0
        uv.x *= float(iResolution.x) / iResolution.y
        rd = ti.Vector([uv.x, uv.y, -2.0])
        rd = normalize(rd)
        
        ro = ti.Vector([0.0, 0.0, 2.5])
        ro = rotateY(ro, roty)
        rd = rotateY(rd, roty)
        color = scene(ro, rd)
        pixels[i, j] = ti.Vector([color.x, color.y, color.z])
    
    # test perlin noise
    # for i, j in pixels:
    #    uv = 1.0 * ti.Vector([i, j]) / iResolution
    #    color = noiseColor.evalColor(uv.x, uv.y, 0.0)
    #    pixels[i, j] = ti.Vector([color.x, color.y, color.z])

gui = ti.GUI("Fur Simulation", res=(n * 2, n))

if __name__ == "__main__":
    noiseAlpha.reset()
    noiseDepth.reset()
    noiseColor.reset()
    iTime[None] = 0.0
    for i in range(3000):
        iTime[None] += 1.0 / 35
        paint()
        gui.set_image(pixels)
        filename = f'frame_{i:05d}.png'
        print(f'Frame {i} is recorded in {filename}')
        gui.show(filename)
