# -*- coding: utf-8 -*-
"""Basic import and functionality tests."""

try:
    import pytest
except ImportError:
    # Create a minimal pytest replacement for standalone testing
    class MockPytest:
        @staticmethod
        def fail(msg):
            raise AssertionError(msg)
        
        @staticmethod
        def skip(msg):
            print("SKIP:", msg)
            return
    
    pytest = MockPytest()

def test_basic_imports():
    """Test that basic imports work correctly."""
    try:
        from temporal_optimizer import StableAdam, StableSGD
        assert StableAdam is not None
        assert StableSGD is not None
    except ImportError as e:
        pytest.fail("Failed to import optimizers: {}".format(str(e)))


def test_basic_optimizer_creation():
    """Test that optimizers can be created."""
    try:
        import torch
        from temporal_optimizer import StableAdam, StableSGD
        
        # Create a simple model
        model = torch.nn.Linear(10, 2)
        
        # Test StableAdam creation
        optimizer_adam = StableAdam(model.parameters())
        assert optimizer_adam is not None
        
        # Test StableSGD creation  
        optimizer_sgd = StableSGD(model.parameters())
        assert optimizer_sgd is not None
        
    except ImportError:
        pytest.skip("PyTorch not available, skipping optimizer creation test")
    except Exception as e:
        pytest.fail("Failed to create optimizers: {}".format(str(e)))


def test_basic_optimization_step():
    """Test that optimizers can perform basic optimization steps."""
    try:
        import torch
        from temporal_optimizer import StableAdam
        
        # Create model and data
        model = torch.nn.Linear(10, 2)
        optimizer = StableAdam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        
        # Basic optimization step
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # If we get here without exceptions, the test passes
        assert True
        
    except ImportError:
        pytest.skip("PyTorch not available, skipping optimization test")
    except Exception as e:
        pytest.fail("Optimization step failed: {}".format(str(e)))


if __name__ == "__main__":
    test_basic_imports()
    test_basic_optimizer_creation() 
    test_basic_optimization_step()
    print("All basic tests passed!")