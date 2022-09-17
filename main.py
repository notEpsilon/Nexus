from nexus.autograd.tensor import Tensor


def main() -> None:
    x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    w = Tensor([[0.5, -0.5], [1, 2]])
    y = Tensor([[0.0], [1.0], [1.0], [0.0]])

    out = x.linear(w)

    loss = ((out - y) ** 2).mean()
    print(f"loss = {loss}")

    loss.backward()

    print(f"x.grad: \n{x.grad}\n")
    print(f"w.grad: \n{w.grad}\n")


if __name__ == "__main__":
    main()
