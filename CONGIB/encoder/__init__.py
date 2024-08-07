import encoder.gin_encoder as gin_encoder
import encoder.gcn_encoder as gcn_encoder
import encoder.edge_gcn_encoder as edge_gcn_encoder
import encoder.mlp as mlp

GINEncoder = gin_encoder.GINEncoder
VGINEncoder = gin_encoder.VGINEncoder

GCNEncoder = gcn_encoder.GCNEncoder
VGCNEncoder = gcn_encoder.VGCNEncoder

EdgeGCNEncoder = edge_gcn_encoder.EdgeGCNEncoder

FNN = mlp.FNN