# 从 C/C++ 到 Haskell——函数式编程设计导引

严昌浩

教材：

- [Real World Haskell](http://cnhaskell.com/)
- [Haskell 趣学指南](https://www.kancloud.cn/kancloud/learnyouahaskell/44651)

## 2019-09-10

C 语言的抽象工具：

- 结构体（`struct`）

  > 程序 = 数据结构 + 算法

  `struct` + function：抽象数据类型（ADT）

- 函数
- 库

  暴露接口，隐藏函数实现细节

- 宏（`#define`）

  Token 替换，无类型

Java/C++ 面向对象分析 + 设计（动词、名词分析法）

- 类：名词（「管理者」）
- 函数：动作（「管理」）

编程范式：

- 过程式：C、汇编
- 面向对象：C++、Java
- 动态语言：Python、Ruby
- **函数式**：Haskell

特点：

- 无变量、无循环、无顺序
- 有函数定义、调用、`if`

## 2019-09-17

神经网络：

- 模拟人类的「快思考」

未来语言的特点：

- 计算资源丰富，性能不再至关重要
- 语言内在简洁
- 不能过早优化
- 开发效率高（行数少）
- 并行

### 递归法数组求和

C 代码：

```c
  Sum(a[1 .. n])
= Sum(a[1 .. n-1]) + a[n]
```

Haskell 代码：

```haskell
Sum :: [Int] -> Int     -- 或任意类型 Sum :: [a] -> a
Sum [] = 0              -- 递归基
Sum x:xs = x + Sum xs   -- 递归
```

### 快速排序

```haskell
qsort [] = []
qsort x:xs = qsort less ++ [x] ++ qsort more
    where less = [y | y <- xs, y <  x]
          more = [y | y <- xs, y >= x]
```

### 类型推导规则

```haskell
    f :: A -> B, e :: A
==> f e :: B
```

### 测试——规格说明

```haskell
prop_Max :: Int -> Int -> Bool
prop_Max x y = (x <= max x y) && (y <= max x y)
```

QuickCheck

## 2019-09-24

```c
x = x + 1  // 左边的 x：左值
           // 右边的 x：右值
```

### 求值顺序：懒惰 + 函数复写

```c
// C
doubleNum (2 + 3) -> doubleNum (5) -> 10
```

```haskell
-- Haskell
doubleNum (2 + 3) -> 2 * (2 + 3) -> 10
```

### 递归——阶乘函数

```haskell
fac :: Int -> Int
fac n
  | n == 0 = 1
  | n >  0 = fac (n - 1) * n
```

### 尾递归

- 最后一个 `return` 语句是单纯函数
- 时间复杂度
  - 尾递归：$O(N)$
  - 普通递归：$O(2N)$
- 空间复杂度：
  - 尾递归：$O(1)$
  - 普通递归：$O(N)$
- 实质是迭代（iteration）

```haskell
sum :: [Int] -> Int -> Int
sum [] n = n
sum [x:xs] n = sum [xs] x + n
```

冯·诺伊曼机：存储程序，顺序执行
分布式计算：数据到齐，点火驱动

```haskell
-- 仅使用 +1 和 -1 实现加法
add :: Int -> Int -> Int
add 0 y = y
add x y = add (x - 1) (y + 1)

-- 仅使用 +1 实现加法
add :: Int -> Int -> Int -> Int
add x y n | x == n = y
add x y n = add x (y + 1) (n + 1)
```

## 2019-10-08

### 优先级 & 结合性

- 函数调用具有最高的优先级
- 函数类型 `->` 是右结合的：
  - `t1 -> t2 -> t3` 等价于 `t1 -> (t2 -> t3)`
  - 但完全不同于 `(t1 -> t2) -> t3`

### 柯里化（Currying）

```haskell
add :: Int -> Int -> Int
add x y = x + y
```

那么

```haskell
add 2 :: Int -> Int
```

### 函数复合

```haskell
(.) :: (a -> b) -> (b -> c) -> (a -> c)
-- 括号可以省略：
(.) :: (a -> b) -> (b -> c) -> a -> c
-- 因此可以定义为：
(.) f g x = g(f x)
-- 也可以写成 lambda：
(.) f g = \ x -> g(f x)
```

实例：

```haskell
quadruple :: Int -> Int
quadruple = double . double  -- Point-free style
-- 等价于：
-- quadruple x = double . double x
```

### 模式匹配

```haskell
shift :: ((Int, Int), Int) -> (Int, (Int, Int))
shift ((x, y), z) = (x, (y, z))
```

### 列表推导

- 列表数字乘 2:

```haskell
doubles :: [Int] -> [Int]
doubles [] = []
doubles (x:xs) = 2 * x : xs
-- 可以改写成：
doubles xs = [2 * x | x <- xs]
```

其中  `<-` 相当于 $\in$

- 求 1 至 $n$ 平方和

```c
int sum = 0;
for (int i = 0; i < n; ++i) {
  sum += i * i;
}
```

```haskell
f = f3 . f2 . f1
f1 n = 1..n
f2 xs = map (^ 2) xs
f3 xs = sum xs
```

### 列表过滤

```haskell
factors :: Int -> [Int]
factors n = [i | i <- [1..n], n `mod` i == 0]
```

列表并不是数组，而是一级级 `:` 组成的（stream）

惰性计算：解耦编程时的顺序与运行时的顺序

## 2019-10-15

不能修改数据时：返回新的数据

### Lisp 语言

- 公理：

  - `cons a b = (a, b)`
  - `car (a, b) = a`
  - `cdr (a, b, c) = (b, c)`

- 实现：

  ```lisp
  (define (cons a b)
    (lambda (pick)
      (cond ((= pick 1) a)
            ((= pick 2) b))))

  (define (car x) (x 1))  ; `x` is a lambda
  (define (cdr x) (x 2))
  ```

- 举例：

  ```lisp
    (car cons 36 49)
  = (car (lambda (pick)
            (cond ((= pick 1) 37)
                  ((= pick 2) 49))))
  = ((lambda (pick)
        (cond ((= pick 1) 37)
              ((= pick 2) 49))) 1)
  = (cond ((= 1 1) 37)
          ((= 1 2) 49))
  = 37
  ```

### 立即计算

```haskell
eval :: Int -> Int
eval 0 = 0
eval e = e  -- 会判断是不是 0，因此需要计算
```

```haskell
eval :: a -> a
eval ⊥ = ⊥
eval e = e  -- 会判断是不是 0，因此需要计算
```

### 高阶函数

- Map

  ```haskell
  map :: (a -> b) -> [a] -> [b]
  map f xs = [f x | x <- xs]
  maf f [] = []
  map f x:xs = f x : map f xs
  ```

  ```haskell
  doubleAll xs = map double xs
  isEvenAll xs = map isEven xs

  -- Point-free atyle:
  doubleAll = map double
  isEvenAll = map isEven
  ```

- Fold

  ```haskell
  foldr :: (a -> b -> b) -> b -> [a] -> b
  -- Type of `z` is `b`
  foldr op z [] = z
  foldr op z (x:xs) = x `op` foldr op z x xs
  ```

  ```haskell
  sum xs = foldr plus 0 xs

  -- Point-free atyle:
  sum = foldr plus 0
  ```

## 2019-10-22

### fold

```haskell
foldr op z [] = z
foldr op z (x:xs) = x `op` foldr op z xs
```

本质是把列表中的 `:` 换成 `` `op` ``

尾递归版本：

```haskell
foldl op z [] = z
foldl op z (x:xs) = foldl (z `op` x) `op` xs
```

然而 `foldr` 可以代替 `foldl`，反之不可

「全速版本」：

```haskell
foldl' op z [] = z
foldl' op z (x:xs) = z' `seq` foldl' (z' `op` xs)
    where z' = z `op` x
```

### 类

- 类（类型类，type class）是类型的集合
- 语义上等价于 Java 的 `interface`

多态：

- 参数多态

  ```haskell
  length :: [a] -> Int
  ```

  相当于 C++ 里面的 `template` 泛型

- *ad hoc* 多态

  ```haskell
  (==) :: a -> a -> Bool
  ```

## 2019-10-29

### 类型类

```haskell
class Eq a where
  (==) : a -> a -> Bool
  (/=) : a -> a -> Bool

instance Eq Bool where
  True  == True  = True
  False == False = True
  _     == _     = False
  x /= y = not (x == y)

elem :: Eq a => a -> [a] -> Bool  -- `=>` 表示限定，要求类型 `a` 是类 `Eq` 的特例
elem x [] = False
elem x (y:ys) = (x == y || elem x ys)
```

缺省定义

```haskell
class Eq a where
  (==), (/=) : a -> a -> Bool
  x /= y = not (x == y)
  x /= y = not (x /= y)
```

### `sum`：实现任意多参数的求和

类型上的递归

目标

```haskell
mySum :: Int -> Int
-- mySum 1 = 1
mySum :: Int -> Int -> Int
-- mySum 1 2 = 3
mySum :: Int -> Int -> Int-> Int
-- mySum 1 2 5 = 8
```

分析

```haskell
mySum :: R r => Int -> r
-- r = Int
--     Int -> Int
--     Int -> Int -> Int...
```

$\forall r\in R$, $\mathrm{Int}\in R$, $(\mathrm{Int}\to r)\in R$

```haskell
class R r where
  mySum :: Int -> r

instance R Int where
  mySum :: Int -> Int
  mySum x = x

instance R (Int -> r) where
  mySum :: Int -> (Int -> r)
  -- mySum x y = x + y 不正确，因为 x+y 是 Int 而非 r
  mySum x y = mySum (x + y)
```

示例

```haskell
   mySum 1 2 3      -- mySum :: Int -> (Int -> r)
-> mySum (1+2) 3    -- mySum :: Int -> (Int -> r)
-> mySum ((1+2)+3)  -- mySum :: Int -> Int
-> ((1+2)+3)
-> 6
```

思考：可否使用 `instance R (Int -> r)`？

```haskell
class R r where
  mySum :: Int -> r

instance R Int where
  mySum :: Int -> Int
  mySum x = x

instance R (r -> Int) where
  mySum :: (r -> Int) -> Int
  mySum x y = <不能实现>
```

## 2019-11-09

### 定义新类型（代数数据类型）

```haskell
data Day = Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
  deriving (Eq, Show)
```

- 左边：类型构造符（type constructor），大写字母开头
- 右边：值构造符（value constructor），大写字母开头
- 默认生成 `==` 和 `show`
- 加法类型

```haskell
data BookInfo = Book Int String [String]
  deriving (Show)
```

- 左边：类型构造符
- 右边：值构造符
  - `Book`：类似 C++ 构造函数
  - `Int String [String]`：占位符
- 乘法类型

如果只有一个 field，可以用 newtype

```haskell
newtype Okay = ExactlyOne Int
```

### `data` 示例

列表

```haskell
data List a = Nil                -- a: 类型变量
              | Cons a (List a)  -- Cons: 前导符
  deriving (Eq, Show)
```

二叉树

```c
struct Node {
  int info;
  Node left, right;
};
```

```haskell
data Tree a = Nil
              | Node a (Tree a) (Tree a)
  deriving (Eq, Show)
```

求深度（递归）

```haskell
depth :: Tree a -> Int
depth Nil = 0
depth (Node n tree1 tree2) = 1 + max (depth tree1) (depth tree2)  -- 使用值构造符
```

### 记录语法（Record syntax）

```haskell
data Book = Book Int String [String]
  deriving (Show)

bookID :: Book -> Int
bookID (Book id _ _) = id

bookName :: Book -> String
bookName (Book _ name _) = name

bookAuthors :: Book -> [String]
bookAuthors (Book _ _ authors) = authors
```

可改写为

```haskell
data Book = Book {
  bookID      :: Int,
  bookName    :: String,
  bookAuthors :: [String],
} deriving (Show)
```

### 示例：算术表达式

表达式类型：

```haskell
data Expr =
    Num Int        -- 数字
  | Add Expr Expr  -- 表达式相加
  | Mul Expr Expr  -- 表达式相乘
```

求值：

```haskell
eval :: Eval -> Int
eval (Num x) = x
eval (Add expr1 expr2) = eval expr1 + eval expr2
eval (Mul expr1 expr2) = eval expr1 * eval expr2
```

### 函子（`Functor`）类型类

| Value               | Type    | Kind     |
|:-------------------:|:--------:|:--------:|
| `3`                 | `Int`    | `*`      |
| `"Hello"`           | `String` | `*`      |
| `[x, y]`            | `List`   | `* -> *` |
| `Nothing \| Just x` | `Maybe`  | `* -> *` |

`*`: contrete type

```haskell
class Functor f where
  fmap :: (a -> b) -> f a -> f b

instance Functor [] where
  fmap = map
```

## 2019-11-12

### 编译期 `class` 消失

类似 C++ 虚函数表

```haskell
square :: Num n => n -> n
square x = x * x

class Num a where
  (+) :: a -> a -> a
  (*) :: a -> a -> a
  negate :: a -> a
  ...
```

编译后转化为

```haskell
square :: Num n -> n -> n
square d x = (*) d x x

-- 变为函数表
data Num a = MakeNum
  (a -> a -> a)
  (a -> a -> a)
  (a -> a)
  ...
```

### IO

动作类型：`IO a`

- 基本 IO 动作：

  ```haskell
  -- 读一个字符
  getChar :: IO Char
  -- 打印字符
  putChar :: IO ()
  -- 返回包裹后的值
  return :: a -> IO a
  ```

- `do` 语句块

  ```haskell
  do {e}             = e
  do {e; stmts}      = e >> do {stmts}
  do {p <- e; stmts} = let ok p = do {stmts}
                           ok _ = fail "..."
                       in e >>= ok  -- 执行 e，脱掉 IO 帽子，再传给 ok
  ```

- `sequence`

  函数原型：

  ```haskell
  sequence :: [IO a] -> IO [a]
  ```

  示例：

  ```haskell
  sequence (map print [1,2,3,4,5])

  1
  2
  3
  4
  5
  [(),(),(),(),()]
  ```

  实现：

  ```haskell
  -- 不带结果：
  sequence ms = foldr (>>) (return []) ms
  -- 带结果：
  sequence ms = foldr k (return []) ms
    where
      k m m' = do
        x  <- m
        xs <- m'
        return (x:xs)
  ```

- `map` 的 IO 版本：

  ```haskell
  mapM print [1,2,3]

  1
  2
  3
  [(),(),()]

  mapM_ print [1,2,3]

  1
  2
  3
  ```

- `forever`

```haskell
forever :: IO a -> IO b
```

## 2019-11-19

### Monad 简介

不存在以下函数：

```haskell
runIO :: IO a -> a
```

`do` 语句块相当于语法糖：

```haskell
do e        = e

do          = e >>= (\x -> do c)
  x <- e
  c
```

绑定操作符 `>>=`，相当于 C 语言中的分号：

```haskell
(>>=): IO a -> (a -> IO b) -> IO b
```

### Monad 类型类

```haskell
class Monad m where
  return :: a -> m a                  -- m 的 kind 是 * -> *
  (>>=)  :: m a -> (a -> m b) -> m b  -- 能够具有连接性
  (>>)   :: m a -> m b -> m b

instance Monad Maybe
instance Monad IO
...
```

示例：

- Identity monad

  ```haskell
  newtype Id a = Id a

  instance Monad Id where
    return x = Id x
    Id x >>= f = f x
  ```

- Maybe monad

  ```haskell
  data Maybe a = Nothing | Just a

  instance Monad Maybe where
    return x = Just x

    (>>=): Maybe a -> (a -> Maybe b) -> Maybe b
    Nothing >>= f = Nothing
    Just x  >>= f = f x  -- 使用 pattern matching 提取值
  ```

- List monad

  ```haskell
  instance Monad [] where
    return x = [x]
    list >>= f = concat (map f list)
  ```

  列表推导（list comprehension）中的 `<-` 相当于 monad 中抽取所有的值

### Monadic 函数复合

```haskell
f :: a -m-> b
f :: a -> m b  -- Monad
f :: m a -> b  -- Comonad
```

普通函数：

```haskell
f :: a -> b
g :: b -> c
h :: a -> c = g . f
```

其中

```haskell
(.) :: (b -> c) -> (a -> b) -> (a -> c)
```

Monadic 函数（带副作用）：

```haskell
f :: a -> m b
g :: b -> m c
h :: a -> m c  -- 不能写成 g . f
```

「重新定义」函数复合：

```haskell
(m.) :: (a -> m b) -> (b -> m c) -> (a -> m c)
g `(m.)` f = \x -> g (f x)
```

### Monad 三定律

```haskell
-- Identity law
1. (return x) >>= f == f x
2. mx >>= return    == mx  -- mx 是类型 m x 的值
-- Associative law
3. (mx >>= f) >>= g == mx >>= (\x -> f x >>= g)
```

编译器不能保证以上语义

---

作业：

- 实现 maybe monad
- 返回缺少 father/mother 的位置

## 2019-12-03

### State monad

实现「全局变量」

```haskell
data State s a = State  s -> (a, s)
-- State 的 kind 是 * -> * -> *
-- 对应的 monad 是 State s (kind 是 * -> *)

impl Monad (State s)
  return :: a -> (State s) a
  return x = State (\s -> (x, s))

  (>>=) :: State s a -> (a -> State s b) -> State s b
  (>>=) (State h) f = State (\s ->
    let (a, s') = h s
        (State g) = f a
    in g s')
```

### GUI

- 本质是另一种编程范式（reactive）
- 回调函数（callback）
  - CPU 中断
  - `getMessage`
