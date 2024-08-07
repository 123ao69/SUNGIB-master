import utils.view_generator as view_generator1
import utils.eva_emb as eva_emb
import utils.attack_edge as attack_edge
import utils.etv_transform as etv_transform

view_generator = view_generator1.view_generator
node_view_generator = view_generator1.node_view_generator
sample_subgraph = view_generator1.sample_subgraph

logger = eva_emb.logger
test_logger = eva_emb.test_logger
eval_node = eva_emb.eval_node
test_classify = eva_emb.test_classify
evaluate_embedding = eva_emb.evaluate_embedding

attack_edge = attack_edge.attack_edge

connective_node_generation = etv_transform.connective_node_generation
connective_graph_generation = etv_transform.connective_graph_generation
