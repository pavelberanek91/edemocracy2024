import agentpy as ap
import matplotlib.pyplot as plt
import seaborn as sns

class Person(ap.Agent):

    def setup(self):
        self.grid = self.model.grid
        self.random = self.model.random
        self.group = self.random.choice(range(self.p.n_groups))
        self.share_similar = 0
        self.happy = False

    def update_happiness(self):
        # zjisti počet agentů se stejným názorem v sousedství, buď štastný, pokud počet stejně smýšlejících sousedů je 
        # podle tvého očekávání
        neighbors = self.grid.neighbors(self)
        similar = len([n for n in neighbors if n.group == self.group])
        ln = len(neighbors)
        self.share_similar = similar / ln if ln > 0 else 0
        self.happy = self.share_similar >= self.p.want_similar

    def find_new_home(self):
        """ Move to random free spot and update free spots. """
        new_spot = self.random.choice(self.model.grid.empty)
        self.grid.move_to(self, new_spot)


class SegregationModel(ap.Model):

    def setup(self):

        # parametry
        s = self.p.size
        n = self.n = int(self.p.density * (s ** 2))

        # vytvoř přížku s agenty
        self.grid = ap.Grid(self, (s, s), track_empty=True)
        self.agents = ap.AgentList(self, n, Person)
        self.grid.add_agents(self.agents, random=True, empty=True)

    def update(self):
        # aktualizuj spokojenost agentů a spočítej počet nespokojených, kteří se ještě budou chtít stěhovat
        self.agents.update_happiness()
        self.unhappy = self.agents.select(self.agents.happy == False)

        # zastav simulaci pokud je dosaženo maximální spokojenosti
        if len(self.unhappy) == 0:
            self.stop()

    def step(self):
        # přesune nespokojené agenty na nová místa
        self.unhappy.find_new_home()

    def get_segregation(self):
        # spočítá poměr sousedů se stejným názorem
        return round(sum(self.agents.share_similar) / self.n, 2)

    def end(self):
        # změř míru segregace na konci simulace
        self.report('segregation', self.get_segregation())

def animation_plot(model, ax):
    group_grid = model.grid.attr_grid('group')
    ap.gridplot(group_grid, cmap='RdBu', ax=ax, alpha=0.8)
    ax.set_title(f"Segregation model \n Time-step: {model.t}, " f"Segregation: {model.get_segregation()}")


def main():
    parameters = {
        'want_similar': 0.7,    # poměr sousedů se stejným názorem pro usazení agenta
        'n_groups': 2,          # počet skupin s rozdílným názorem
        'density': 0.95,        # hustota populace
        'size': 50,             # šířka a výška simulační mřížky
        'steps': 50             # maximální počet kroků simulace
    }    
    
    fig, ax = plt.subplots()
    model = SegregationModel(parameters)
    animation = ap.animate(model, fig, ax, animation_plot)
    animation.save('segregation.gif', writer='imagemagick', fps=5)


if __name__ == '__main__':
    main()