"""Core data models for the Vietnamese Fact-Checking System."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import numpy as np


@dataclass
class Claim:
    """Represents a factual claim to be verified."""
    
    text: str
    context: str = ""
    confidence: float = 0.0
    sentence_type: str = "factual_claim"
    start_idx: int = 0
    end_idx: int = 0
    language: str = "vi"
    id: Optional[str] = None
    
    def __post_init__(self):
        """Validate claim data after initialization."""
        if not self.text:
            raise ValueError("Claim text cannot be empty")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        
        if self.sentence_type not in ["factual_claim", "opinion", "question", "command"]:
            raise ValueError(f"Invalid sentence_type: {self.sentence_type}")
        
        if self.start_idx < 0 or self.end_idx < 0:
            raise ValueError("Indices cannot be negative")
        
        if self.start_idx > self.end_idx:
            raise ValueError("start_idx cannot be greater than end_idx")
        
        # Generate ID if not provided
        if self.id is None:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for claim."""
        import hashlib
        content = f"{self.text}_{self.start_idx}_{self.end_idx}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert claim to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        """Create claim from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert claim to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Claim":
        """Create claim from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class Evidence:
    """Represents a piece of evidence for claim verification."""
    
    text: str
    source_url: str
    source_title: str = ""
    source_author: Optional[str] = None
    publish_date: Optional[str] = None  # ISO format string
    credibility_score: float = 0.0
    language: str = "vi"
    stance: Optional[str] = None
    stance_confidence: Optional[float] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        """Validate evidence data after initialization."""
        if not self.text:
            raise ValueError("Evidence text cannot be empty")
        
        if not self.source_url:
            raise ValueError("Source URL cannot be empty")
        
        if not 0 <= self.credibility_score <= 1:
            raise ValueError(f"Credibility score must be in [0, 1], got {self.credibility_score}")
        
        if self.stance is not None and self.stance not in ["support", "refute", "neutral"]:
            raise ValueError(f"Invalid stance: {self.stance}")
        
        if self.stance_confidence is not None and not 0 <= self.stance_confidence <= 1:
            raise ValueError(f"Stance confidence must be in [0, 1], got {self.stance_confidence}")
        
        # Generate ID if not provided
        if self.id is None:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for evidence."""
        import hashlib
        content = f"{self.source_url}_{self.text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        """Create evidence from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert evidence to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Evidence":
        """Create evidence from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class ReasoningStep:
    """One step in the ReAct loop."""
    
    iteration: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    timestamp: Optional[str] = None  # ISO format string
    
    def __post_init__(self):
        """Validate reasoning step data after initialization."""
        if self.iteration < 0:
            raise ValueError("Iteration must be non-negative")
        
        if not self.thought:
            raise ValueError("Thought cannot be empty")
        
        if not self.action:
            raise ValueError("Action cannot be empty")
        
        if not self.observation:
            raise ValueError("Observation cannot be empty")
        
        # Set timestamp if not provided
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reasoning step to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningStep":
        """Create reasoning step from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert reasoning step to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ReasoningStep":
        """Create reasoning step from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class Verdict:
    """Final verdict on claim verification."""
    
    claim_id: str
    label: str
    confidence_scores: Dict[str, float]
    supporting_evidence: List[str] = field(default_factory=list)
    refuting_evidence: List[str] = field(default_factory=list)
    explanation: str = ""
    reasoning_trace: List[ReasoningStep] = field(default_factory=list)
    quality_score: float = 0.0
    
    def __post_init__(self):
        """Validate verdict data after initialization."""
        if not self.claim_id:
            raise ValueError("Claim ID cannot be empty")
        
        if self.label not in ["supported", "refuted", "not_enough_info"]:
            raise ValueError(f"Invalid label: {self.label}")
        
        # Validate confidence scores
        if not self.confidence_scores:
            raise ValueError("Confidence scores cannot be empty")
        
        for label, score in self.confidence_scores.items():
            if not 0 <= score <= 1:
                raise ValueError(f"Confidence score for {label} must be in [0, 1], got {score}")
        
        # Check that confidence scores sum to approximately 1.0
        total = sum(self.confidence_scores.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Confidence scores must sum to 1.0, got {total}")
        
        if not 0 <= self.quality_score <= 1:
            raise ValueError(f"Quality score must be in [0, 1], got {self.quality_score}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert verdict to dictionary."""
        data = asdict(self)
        # Convert reasoning trace to list of dicts
        data['reasoning_trace'] = [step.to_dict() for step in self.reasoning_trace]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Verdict":
        """Create verdict from dictionary."""
        # Convert reasoning trace from list of dicts
        if 'reasoning_trace' in data and data['reasoning_trace']:
            data['reasoning_trace'] = [
                ReasoningStep.from_dict(step) if isinstance(step, dict) else step
                for step in data['reasoning_trace']
            ]
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert verdict to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Verdict":
        """Create verdict from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class GraphNode:
    """Node in the reasoning graph."""
    
    id: str
    type: str
    text: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate graph node data after initialization."""
        if not self.id:
            raise ValueError("Node ID cannot be empty")
        
        if self.type not in ["entity", "claim", "evidence"]:
            raise ValueError(f"Invalid node type: {self.type}")
        
        if not self.text:
            raise ValueError("Node text cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph node to dictionary."""
        data = asdict(self)
        # Convert numpy array to list for JSON serialization
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        """Create graph node from dictionary."""
        # Convert embedding list back to numpy array
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)


@dataclass
class GraphEdge:
    """Edge in the reasoning graph."""
    
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    evidence_source: str = ""
    
    def __post_init__(self):
        """Validate graph edge data after initialization."""
        if not self.source_id:
            raise ValueError("Source ID cannot be empty")
        
        if not self.target_id:
            raise ValueError("Target ID cannot be empty")
        
        if not self.relation:
            raise ValueError("Relation cannot be empty")
        
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be in [0, 1], got {self.weight}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph edge to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdge":
        """Create graph edge from dictionary."""
        return cls(**data)


@dataclass
class KnowledgeGraph:
    """Dynamic reasoning graph for fact-checking."""
    
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)
    
    def add_node(self, node: GraphNode) -> None:
        """Add or update a node in the graph."""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        # Validate that source and target nodes exist
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_id} not found in graph")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_id} not found in graph")
        
        self.edges.append(edge)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edges_from(self, node_id: str) -> List[GraphEdge]:
        """Get all edges originating from a node."""
        return [edge for edge in self.edges if edge.source_id == node_id]
    
    def get_edges_to(self, node_id: str) -> List[GraphEdge]:
        """Get all edges pointing to a node."""
        return [edge for edge in self.edges if edge.target_id == node_id]
    
    def node_count(self) -> int:
        """Get number of nodes in graph."""
        return len(self.nodes)
    
    def edge_count(self) -> int:
        """Get number of edges in graph."""
        return len(self.edges)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert knowledge graph to dictionary."""
        return {
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': [edge.to_dict() for edge in self.edges]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Create knowledge graph from dictionary."""
        graph = cls()
        
        # Reconstruct nodes
        if 'nodes' in data:
            for node_id, node_data in data['nodes'].items():
                graph.nodes[node_id] = GraphNode.from_dict(node_data)
        
        # Reconstruct edges
        if 'edges' in data:
            graph.edges = [GraphEdge.from_dict(edge_data) for edge_data in data['edges']]
        
        return graph
    
    def to_json(self) -> str:
        """Convert knowledge graph to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgeGraph":
        """Create knowledge graph from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class FactCheckResult:
    """Complete fact-checking result."""
    
    claim: Claim
    verdict: Verdict
    evidence: List[Evidence]
    reasoning_graph: KnowledgeGraph
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fact-check result to dictionary."""
        return {
            'claim': self.claim.to_dict(),
            'verdict': self.verdict.to_dict(),
            'evidence': [ev.to_dict() for ev in self.evidence],
            'reasoning_graph': self.reasoning_graph.to_dict(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactCheckResult":
        """Create fact-check result from dictionary."""
        return cls(
            claim=Claim.from_dict(data['claim']),
            verdict=Verdict.from_dict(data['verdict']),
            evidence=[Evidence.from_dict(ev) for ev in data['evidence']],
            reasoning_graph=KnowledgeGraph.from_dict(data['reasoning_graph']),
            metadata=data.get('metadata', {})
        )
    
    def to_json(self) -> str:
        """Convert fact-check result to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "FactCheckResult":
        """Create fact-check result from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
