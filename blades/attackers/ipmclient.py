from blades.clients.client import ByzantineClient
from typing import Generator, List
from blades.clients import BladesClient

class IpmClient(ByzantineClient):
    def __init__(self, epsilon: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def omniscient_callback(self, clients: List[BladesClient]):
        pass

class IpmAdversary:
    def __init__(self, epsilon: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def omniscient_callback(self, clients: List[BladesClient]):
        updates = []
        for client in clients:
            if client.is_byzantine():
                updates.append(client.get_update())

        update = -self.epsilon * (sum(updates)) / len(updates)
        for client in clients:
            if client.is_byzantine():
                client.save_update(update)
