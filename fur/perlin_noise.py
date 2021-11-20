import taichi as ti

@ti.func
def rand3():
    return ti.Vector([ti.random(), ti.random(), ti.random()])

@ti.func
def smoothstep(x):
    return x * x * (3.0 - 2.0 * x)

@ti.func
def lerp(x, y, t):
    return x * (1.0 - t) + y * t

@ti.func
def length(v):
    l = ti.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    return l

@ti.func
def normalize(v):
    l = length(v)
    v /= l
    return v

@ti.data_oriented
class PerlinNoise:
    def __init__(self, frequency, amplitude):
        self.table_size = 256
        self.frequency = frequency
        self.amplitude = amplitude
        self.table_size_mask = self.table_size - 1
        self.phase1 = ti.field(ti.f32, (3))
        self.phase2 = ti.field(ti.f32, (3))
        self.gradients = ti.Vector.field(3, dtype=ti.f32, shape=(self.table_size))
        self.permutation_table = ti.field(ti.i32, (self.table_size * 2))

    @ti.kernel
    def reset(self):
        for i in range(3):
            self.phase1[i] = ti.random() / self.frequency
            self.phase2[i] = ti.random() / self.frequency
        for i in range(self.table_size):
            self.gradients[i] = 2.0 * rand3() - ti.Vector([1.0, 1.0, 1.0])
            while length(self.gradients[i]) >= 1.0:
                self.gradients[i] = 2.0 * rand3() - ti.Vector([1.0, 1.0, 1.0])
            self.gradients[i] = normalize(self.gradients[i])
            self.permutation_table[i] = i

        for i in range(self.table_size):
            index = int(ti.random() * self.table_size)
            temp = self.permutation_table[i]
            self.permutation_table[i] = self.permutation_table[index]
            self.permutation_table[index] = temp

        for i in range(self.table_size):
            self.permutation_table[i + self.table_size] = self.permutation_table[i]

    @ti.func
    def hash(self, x, y, z):
        return self.permutation_table[self.permutation_table[self.permutation_table[x] + y] + z]

    @ti.func
    def eval(self, x, y, z):
        x *= self.frequency
        y *= self.frequency
        z *= self.frequency

        xi0 = int(ti.floor(x)) & self.table_size_mask
        yi0 = int(ti.floor(y)) & self.table_size_mask
        zi0 = int(ti.floor(z)) & self.table_size_mask

        xi1 = (xi0 + 1) & self.table_size_mask
        yi1 = (yi0 + 1) & self.table_size_mask
        zi1 = (zi0 + 1) & self.table_size_mask

        tx = x - xi0
        ty = y - yi0
        tz = z - zi0

        u = smoothstep(tx)
        v = smoothstep(ty)
        w = smoothstep(tz)

        # gradients at the corner of the cell
        c000 = self.gradients[self.hash(xi0, yi0, zi0)]
        c100 = self.gradients[self.hash(xi1, yi0, zi0)]
        c010 = self.gradients[self.hash(xi0, yi1, zi0)]
        c110 = self.gradients[self.hash(xi1, yi1, zi0)]

        c001 = self.gradients[self.hash(xi0, yi0, zi1)]
        c101 = self.gradients[self.hash(xi1, yi0, zi1)]
        c011 = self.gradients[self.hash(xi0, yi1, zi1)]
        c111 = self.gradients[self.hash(xi1, yi1, zi1)]

        
        x0 = tx
        x1 = tx - 1.0
        y0 = ty
        y1 = ty - 1.0
        z0 = tz
        z1 = tz - 1.0

        p000 = ti.Vector([x0, y0, z0])
        p100 = ti.Vector([x1, y0, z0])
        p010 = ti.Vector([x0, y1, z0])
        p110 = ti.Vector([x1, y1, z0])

        p001 = ti.Vector([x0, y0, z1])
        p101 = ti.Vector([x1, y0, z1])
        p011 = ti.Vector([x0, y1, z1])
        p111 = ti.Vector([x1, y1, z1])

        a = lerp(c000.dot(p000), c100.dot(p100), u)
        b = lerp(c010.dot(p010), c110.dot(p110), u)
        c = lerp(c001.dot(p001), c101.dot(p101), u)
        d = lerp(c011.dot(p011), c111.dot(p111), u)

        e = lerp(a, b, v)
        f = lerp(c, d, v)

        g = lerp(e, f, w)
        g = (g + 1.0) / 2.0
        return g * self.amplitude

    @ti.func
    def evalColor(self, x, y, z):
        r = self.eval(x, y, z)
        g = self.eval(x + self.phase1[0], y + self.phase1[1], z + self.phase1[2])
        b = self.eval(x + self.phase2[0], y + self.phase2[1], z + self.phase2[2])
        return ti.Vector([r, g, b])


