"""Knowledge graph builder for fact-checking reasoning."""

import logging
from typing import List, Dict, Optional, Set, Tuple
import hashlib
import numpy as np
from difflib import SequenceMatcher

from src.data_models import Evidence, KnowledgeGraph, GraphNode, GraphEdge
from src.ner_extractor import NERExtractor, Entity
from src.relation_extractor import RelationExtractor, Relation

# Setup logging
logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build dynamic reasoning graphs from evidence for fact-checking."""
    
    def __init__(self, entity_similarity_threshold: float = 0.85):
        """Initialize graph builder."""
        self.entity_similarity_threshold = entity_similarity_threshold
        self.ner_extractor = NERExtractor()
        self.relation_extractor = RelationExtractor()
        logger.info(f"GraphBuilder initialized with similarity threshold: {entity_similarity_threshold}")

    
    def build_graph(self, evidence_list: List[Evidence]) -> KnowledgeGraph:
        """Build a knowledge graph from a list of evidence."""
        if not evidence_list:
            logger.warning("Empty evidence list provided to build_graph")
            return KnowledgeGraph()
        
        graph = KnowledgeGraph()
        entity_map: Dict[str, str] = {}
        
        for evidence in evidence_list:
            logger.debug(f"Processing evidence: {evidence.id}")
            entities = self.ner_extractor.extract_entities(text=evidence.text, language=evidence.language)
            
            evidence_node_id = f"evidence_{evidence.id}"
            evidence_node = GraphNode(
                id=evidence_node_id,
                type="evidence",
                text=evidence.text,
                attributes={
                    'source_url': evidence.source_url,
                    'credibility_score': evidence.credibility_score,
                    'language': evidence.language,
                    'stance': evidence.stance,
                    'stance_confidence': evidence.stance_confidence
                }
            )
            graph.add_node(evidence_node)
            
            entity_nodes = []
            for entity in entities:
                canonical_id = self._find_or_create_entity_node(graph=graph, entity=entity, entity_map=entity_map)
                entity_nodes.append(canonical_id)
                mention_edge = GraphEdge(
                    source_id=evidence_node_id,
                    target_id=canonical_id,
                    relation="mentions",
                    weight=entity.confidence,
                    evidence_source=evidence.source_url
                )
                graph.add_edge(mention_edge)
            
            if len(entities) >= 2:
                relations = self.relation_extractor.extract_relations(text=evidence.text, entities=entities, language=evidence.language)
                for relation in relations:
                    subj_canonical = self._get_canonical_id(entity=relation.subject, entity_map=entity_map)
                    obj_canonical = self._get_canonical_id(entity=relation.object, entity_map=entity_map)
                    if subj_canonical and obj_canonical:
                        relation_edge = GraphEdge(
                            source_id=subj_canonical,
                            target_id=obj_canonical,
                            relation=relation.relation_type,
                            weight=relation.confidence,
                            evidence_source=evidence.source_url
                        )
                        graph.add_edge(relation_edge)
            
            if evidence.stance in ["support", "refute"]:
                stance_relation = "supports" if evidence.stance == "support" else "refutes"
                evidence_node.attributes['stance_relation'] = stance_relation
        
        logger.info(f"Built graph with {graph.node_count()} nodes and {graph.edge_count()} edges")
        return graph

    
    def _find_or_create_entity_node(self, graph: KnowledgeGraph, entity: Entity, entity_map: Dict[str, str]) -> str:
        """Find existing entity node or create new one, handling duplicates."""
        normalized_text = entity.text.lower().strip()
        if normalized_text in entity_map:
            return entity_map[normalized_text]
        
        for existing_text, existing_id in entity_map.items():
            if entity.type == graph.get_node(existing_id).attributes.get('entity_type'):
                similarity = self._calculate_text_similarity(normalized_text, existing_text)
                if similarity >= self.entity_similarity_threshold:
                    logger.debug(f"Merging entity '{entity.text}' with existing '{existing_text}' (similarity: {similarity:.2f})")
                    entity_map[normalized_text] = existing_id
                    existing_node = graph.get_node(existing_id)
                    if existing_node:
                        if 'mentions' not in existing_node.attributes:
                            existing_node.attributes['mentions'] = []
                        existing_node.attributes['mentions'].append(entity.text)
                    return existing_id
        
        node_id = self._generate_entity_node_id(entity)
        entity_node = GraphNode(
            id=node_id,
            type="entity",
            text=entity.text,
            attributes={
                'entity_type': entity.type,
                'language': entity.language,
                'confidence': entity.confidence,
                'mentions': [entity.text]
            }
        )
        graph.add_node(entity_node)
        entity_map[normalized_text] = node_id
        logger.debug(f"Created new entity node: {node_id} ({entity.type})")
        return node_id
    
    def _get_canonical_id(self, entity: Entity, entity_map: Dict[str, str]) -> Optional[str]:
        """Get canonical node ID for an entity."""
        normalized_text = entity.text.lower().strip()
        return entity_map.get(normalized_text)
    
    def _generate_entity_node_id(self, entity: Entity) -> str:
        """Generate unique ID for entity node."""
        content = f"{entity.type}_{entity.text}_{entity.language}"
        hash_value = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"entity_{hash_value}"
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if text1 == text2:
            return 1.0
        char_similarity = SequenceMatcher(None, text1, text2).ratio()
        words1 = set(text1.split())
        words2 = set(text2.split())
        if words1 and words2:
            word_similarity = len(words1 & words2) / len(words1 | words2)
        else:
            word_similarity = 0.0
        similarity = 0.6 * char_similarity + 0.4 * word_similarity
        return similarity

    
    def add_claim_node(self, graph: KnowledgeGraph, claim_text: str, claim_id: str) -> str:
        """Add a claim node to the graph and connect it to evidence."""
        claim_node_id = f"claim_{claim_id}"
        claim_node = GraphNode(
            id=claim_node_id,
            type="claim",
            text=claim_text,
            attributes={'claim_id': claim_id}
        )
        graph.add_node(claim_node)
        for node_id, node in graph.nodes.items():
            if node.type == "evidence":
                stance_relation = node.attributes.get('stance_relation')
                if stance_relation:
                    stance_edge = GraphEdge(
                        source_id=node_id,
                        target_id=claim_node_id,
                        relation=stance_relation,
                        weight=node.attributes.get('stance_confidence', 0.5),
                        evidence_source=node.attributes.get('source_url', '')
                    )
                    graph.add_edge(stance_edge)
        logger.info(f"Added claim node: {claim_node_id}")
        return claim_node_id
    
    def handle_contradictions(self, graph: KnowledgeGraph, claim_node_id: str) -> Dict[str, List[str]]:
        """Identify and preserve contradictory evidence in the graph."""
        supporting_evidence = []
        refuting_evidence = []
        edges_to_claim = graph.get_edges_to(claim_node_id)
        for edge in edges_to_claim:
            if edge.relation == "supports":
                supporting_evidence.append(edge.source_id)
            elif edge.relation == "refutes":
                refuting_evidence.append(edge.source_id)
        if supporting_evidence and refuting_evidence:
            logger.info(f"Contradiction detected for claim {claim_node_id}: {len(supporting_evidence)} supporting, {len(refuting_evidence)} refuting")
            claim_node = graph.get_node(claim_node_id)
            if claim_node:
                claim_node.attributes['has_contradiction'] = True
                claim_node.attributes['supporting_count'] = len(supporting_evidence)
                claim_node.attributes['refuting_count'] = len(refuting_evidence)
                support_weight = sum(graph.get_node(ev_id).attributes.get('credibility_score', 0.5) for ev_id in supporting_evidence)
                refute_weight = sum(graph.get_node(ev_id).attributes.get('credibility_score', 0.5) for ev_id in refuting_evidence)
                claim_node.attributes['support_weight'] = support_weight
                claim_node.attributes['refute_weight'] = refute_weight
        return {'supporting': supporting_evidence, 'refuting': refuting_evidence}
    
    def to_dgl_graph(self, graph: KnowledgeGraph):
        """Convert KnowledgeGraph to DGL format for GNN processing."""
        try:
            import dgl
            import torch
        except ImportError:
            logger.error("DGL library not installed. Install with: pip install dgl torch")
            raise ImportError("DGL library required for graph neural network processing")
        if graph.node_count() == 0:
            logger.warning("Cannot convert empty graph to DGL format")
            return None
        node_ids = list(graph.nodes.keys())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        src_nodes = []
        dst_nodes = []
        edge_types = []
        edge_weights = []
        for edge in graph.edges:
            src_idx = node_to_idx[edge.source_id]
            dst_idx = node_to_idx[edge.target_id]
            src_nodes.append(src_idx)
            dst_nodes.append(dst_idx)
            edge_types.append(edge.relation)
            edge_weights.append(edge.weight)
        if src_nodes:
            dgl_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=len(node_ids))
            dgl_graph.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)
            node_types = []
            node_texts = []
            for node_id in node_ids:
                node = graph.get_node(node_id)
                node_types.append(node.type)
                node_texts.append(node.text)
            dgl_graph.ndata['node_id'] = node_ids
            dgl_graph.ndata['node_type'] = node_types
            logger.info(f"Converted to DGL graph: {dgl_graph.num_nodes()} nodes, {dgl_graph.num_edges()} edges")
            return dgl_graph
        else:
            dgl_graph = dgl.graph(([], []), num_nodes=len(node_ids))
            dgl_graph.ndata['node_id'] = node_ids
            logger.warning("Created DGL graph with no edges")
            return dgl_graph


def build_graph_from_evidence(evidence_list: List[Evidence]) -> KnowledgeGraph:
    """Build a knowledge graph from evidence list."""
    builder = GraphBuilder()
    return builder.build_graph(evidence_list)
