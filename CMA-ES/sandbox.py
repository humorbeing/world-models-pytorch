import numpy as np
import cma

def ev(para):
    x = para[0]
    y = para[1]
    z = np.sin(x**2) + np.cos(y**3 + 5)
    return z
para = np.array([0.0, 0.0])
es = cma.CMAEvolutionStrategy(para, 6, inopts={"popsize": 200})
for _ in range(50):
    params_set, fitnesses = es.ask_and_eval(ev)
    es.tell(params_set, fitnesses)
    best_overall_params, best_overall_fitness, _ = es.best.get()
    print('*' * 20)
    # print(generation)
    print(best_overall_fitness)
    print('*' * 20)