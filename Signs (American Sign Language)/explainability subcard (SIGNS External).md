Field                                                                                                  |  Response
:------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------
Intended Task/Domain:                                                                                  |  Detecting sign language in real-time video streams 
Model Type:                                                                                            |  Object Recognition
Intended Users:                                                                                        |  Users of video conferencing applications and those learning American Sign Language
Output:                                                                                                |  Text: instructive feedback for matching hand landmarks' handshape, palm orientation, hand location, and movement.  
Describe how the model works:                                                                          |  Video Frames extract hand and pose landmarks to classify against predefined and validated template of static handshapes, movement, and location. 
Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of:  |  None
Technical Limitations & Mitigation:                                                                    |  Poor lighting, camera angles, distance from camera among other may affect landmark detection accuracy. Some of these limitations can be mitigated by optimizing parameters such as confidence thresholds and frame hold times. 
Verified to have met prescribed NVIDIA quality standards:  |  Yes
Performance Metrics:                                                                                   |  Hand Shape Classification Accuracy, Sign Recognition Accuracy, Latency
Potential Known Risks:                                                                                 |  This model may incorrectly classify or mischaracterize hand location, handshapes, palm orientation, movement, and non-manual signals.
Licensing:                                                                                             |  For Use by NVIDIA & Hello Monday Only

