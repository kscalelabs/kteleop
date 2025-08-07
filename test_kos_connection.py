#!/usr/bin/env python3
"""
Simple test to verify KOS connection and API usage.
"""

import asyncio
import pykos

async def test_kos_connection():
    """Test basic KOS connection and API calls."""
    try:
        print("Testing KOS connection...")
        
        # Test 1: Create KOS client directly
        kos = pykos.KOS()
        print("✅ KOS client created")
        
        # Test 2: Get actuator states (try without await first)
        actuator_ids = [11, 12, 13, 14, 15]  # Left arm
        print(f"Requesting states for actuators: {actuator_ids}")
        
        # Try different approaches
        try:
            # Try 1: Direct call (might be sync)
            state = kos.actuator.get_actuators_state(actuator_ids)
            print("✅ Got response (sync call)")
            print(f"Response type: {type(state)}")
            print(f"Response: {state}")
            
            if hasattr(state, 'states'):
                print(f"✅ Got response with {len(state.states)} actuators")
                for actuator_state in state.states:
                    print(f"  Actuator {actuator_state.actuator_id}: pos={actuator_state.position:.4f}, vel={actuator_state.velocity:.4f}")
            elif isinstance(state, list):
                print(f"✅ Got response with {len(state)} actuators (list)")
                for actuator_state in state:
                    print(f"  Actuator state: {actuator_state}")
            else:
                print(f"Unexpected response type: {type(state)}")
                
        except Exception as e1:
            print(f"Sync call failed: {e1}")
            try:
                # Try 2: Async call
                state = await kos.actuator.get_actuators_state(actuator_ids)
                print("✅ Got response (async call)")
            except Exception as e2:
                print(f"Async call failed: {e2}")
                return
        
        # Test 3: Send a command
        print("\nTesting command...")
        commands = [
            pykos.services.actuator.ActuatorCommand(actuator_id=11, position=0.1)
        ]
        
        try:
            result = kos.actuator.command_actuators(commands)
            print("✅ Command sent successfully (sync)")
            print(f"Command result: {result}")
        except Exception as e:
            try:
                result = await kos.actuator.command_actuators(commands)
                print("✅ Command sent successfully (async)")
                print(f"Command result: {result}")
            except Exception as e2:
                print(f"Command failed: {e2}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_kos_connection()) 