import React, { useState, useEffect } from 'react';
import { UploadCloud, Image as ImageIcon, Video, ArrowLeft, Loader2, CheckCircle2, ChevronRight, Play, Pause, RotateCcw, Crosshair, Fingerprint, Activity, Clock } from 'lucide-react';

const API_BASE = "http://localhost:8000";

function App() {
  const [view, setView] = useState('upload'); // 'upload', 'loading', 'lineup', 'results'
  const [loadingText, setLoadingText] = useState('');
  
  const [sessionId, setSessionId] = useState(null);
  const [thumbnails, setThumbnails] = useState([]);
  const [matches, setMatches] = useState([]);
  const [extractedFeatures, setExtractedFeatures] = useState([]);
  
  const [isImageQuery, setIsImageQuery] = useState(false);
  
  const [playingEvidence, setPlayingEvidence] = useState(null);
  const [evidenceKey, setEvidenceKey] = useState(Date.now());
  const [isEvidencePaused, setIsEvidencePaused] = useState(false);

  // Dynamic Stepper Logic
  const steps = isImageQuery 
    ? [
        { id: 'upload', label: 'Upload Media' },
        { id: 'results', label: 'Analysis Results' }
      ]
    : [
        { id: 'upload', label: 'Upload Media' },
        { id: 'lineup', label: 'Select Target' },
        { id: 'results', label: 'Analysis Results' }
      ];

  const getCurrentStepIndex = () => {
    if (view === 'upload') return 0;
    if (view === 'lineup') return 1;
    if (view === 'results') return isImageQuery ? 1 : 2;
    return 0;
  };

  const jumpToStep = (stepId, index) => {
    if (index > getCurrentStepIndex()) return;
    if (stepId === 'lineup' && !sessionId) return;
    
    setView(stepId);
    if (stepId === 'upload') reset();
    if (stepId === 'lineup') {
      setPlayingEvidence(null);
      setIsEvidencePaused(false);
    }
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setView('loading');
    setLoadingText('Extracting multi-modal features...');
    setIsImageQuery(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/upload/image`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setExtractedFeatures(data.extracted_features);
      setMatches(data.matches);
      setView('results');
    } catch (err) {
      console.error(err);
      alert("Failed to process image.");
      setView('upload');
    }
  };

  const handleVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setView('loading');
    setLoadingText('Tracking subjects in video feed...');
    setIsImageQuery(false);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/upload/video`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setSessionId(data.session_id);
      setThumbnails(data.thumbnails);
      setView('lineup');
    } catch (err) {
      console.error(err);
      alert("Failed to process video.");
      setView('upload');
    }
  };

  const handleTargetSelect = async (trackId) => {
    setView('loading');
    setLoadingText(`Synthesizing identity descriptor for Target ${trackId}...`);

    try {
      const res = await fetch(`${API_BASE}/extract/video/target`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          track_id: trackId
        })
      });
      const data = await res.json();
      setExtractedFeatures(["face", "appearance", "gait"]);
      setMatches(data.matches);
      setView('results');
    } catch (err) {
      console.error(err);
      alert("Failed to extract target.");
      setView('lineup');
    }
  };

  const handlePlayEvidence = (targetId) => {
    if (playingEvidence === targetId) {
      setPlayingEvidence(null);
      setIsEvidencePaused(false);
    } else {
      setEvidenceKey(Date.now());
      setPlayingEvidence(targetId);
      setIsEvidencePaused(false);
    }
  };

  const handleRestartEvidence = () => {
    setEvidenceKey(Date.now());
    setIsEvidencePaused(false);
  };

  const handleTogglePause = () => {
    if (!isEvidencePaused) {
      // Pause: Draw the current frame to the canvas to freeze it
      const canvas = document.getElementById('evidence-canvas');
      const img = document.getElementById('evidence-img');
      if (canvas && img) {
        canvas.width = img.naturalWidth || 640;
        canvas.height = img.naturalHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
      }
      setIsEvidencePaused(true);
    } else {
      // Resume
      setIsEvidencePaused(false);
    }
  };

  const handleBack = () => {
    if (view === 'results' && !isImageQuery && sessionId) {
      // Go back to target lineup
      setView('lineup');
      setPlayingEvidence(null);
      setIsEvidencePaused(false);
    } else {
      reset();
    }
  };

  const formatDateTime = (startSeconds, endSeconds) => {
    const s = Math.max(0, Number(startSeconds) || 0);
    const e = Math.max(s, Number(endSeconds) || s);

    const today = new Date();
    // If timestamp is already seconds from midnight (>= 28800 = 8h), start from 00:00:00
    if (s >= 28800) {
      today.setHours(0, 0, 0, 0);
    } else {
      today.setHours(8, 0, 0, 0);
    }

    const startTime = new Date(today.getTime() + s * 1000);
    const endTime = new Date(today.getTime() + e * 1000);

    const dateStr = startTime.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    const startTimeStr = startTime.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const endTimeStr = endTime.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    return { dateStr, startTimeStr, endTimeStr };
  };

  const reset = () => {
    setView('upload');
    setSessionId(null);
    setThumbnails([]);
    setMatches([]);
    setExtractedFeatures([]);
    setPlayingEvidence(null);
    setIsEvidencePaused(false);
    setIsImageQuery(false);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans flex flex-col relative overflow-x-hidden text-base">
      
      {/* Background ambient glow */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-cyan-900/10 blur-[120px]"></div>
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-blue-900/10 blur-[120px]"></div>
      </div>

      <div className="w-[95%] max-w-[1600px] mx-auto p-4 md:py-8 relative z-10 flex-grow flex flex-col">
        
        {/* Header & Stepper */}
        <header className="mb-8 md:mb-12">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-8">
            <div>
              <h1 className="text-4xl font-bold text-white flex items-center gap-3">
                <Crosshair className="text-cyan-400 w-8 h-8" />
                Surveillance System
              </h1>
              <p className="text-slate-300 mt-2 text-lg">Multi-Modal Subject Retrieval Framework</p>
            </div>
            
            {/* Action Buttons */}
            {view !== 'upload' && view !== 'loading' && (
              <div className="flex gap-3">
                <button onClick={handleBack} className="flex items-center gap-2 text-slate-300 hover:text-white hover:bg-slate-800 transition-all px-4 py-2 rounded-lg border border-slate-700 text-base font-semibold">
                  <ArrowLeft size={18} /> Back
                </button>
                <button onClick={reset} className="flex items-center gap-2 text-slate-300 hover:text-white hover:bg-slate-800 transition-all px-4 py-2 rounded-lg border border-slate-700 text-base font-semibold">
                  <RotateCcw size={18} /> New Search
                </button>
              </div>
            )}
          </div>

          {/* Stepper Navigation */}
          <div className="flex flex-wrap items-center gap-2 text-base font-semibold">
            {steps.map((step, idx) => {
              const isActive = idx === getCurrentStepIndex();
              const isPast = idx < getCurrentStepIndex();
              const isClickable = isPast && view !== 'loading';

              return (
                <React.Fragment key={step.id}>
                  <div 
                    onClick={() => isClickable ? jumpToStep(step.id, idx) : null}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300 ${
                      isActive ? 'bg-cyan-900/40 text-cyan-300 border border-cyan-500/50' :
                      isPast ? 'text-slate-300 hover:text-white hover:bg-slate-800 cursor-pointer border border-transparent hover:border-slate-600' :
                      'text-slate-500 cursor-not-allowed border border-transparent'
                    }`}
                  >
                    <span className={`flex items-center justify-center w-7 h-7 rounded-full text-sm transition-colors duration-300 ${
                      isActive ? 'bg-cyan-500 text-slate-900' : 
                      isPast ? 'bg-slate-600 text-slate-100' : 
                      'bg-slate-800 text-slate-500'
                    }`}>
                      {idx + 1}
                    </span>
                    {step.label}
                  </div>
                  {idx < steps.length - 1 && <ChevronRight size={20} className="text-slate-600" />}
                </React.Fragment>
              );
            })}
          </div>
        </header>

        {/* Main Content Area */}
        <main className="bg-slate-800/80 backdrop-blur-xl rounded-2xl shadow-xl border border-slate-700 p-6 md:p-10 flex-grow flex flex-col relative mb-10">
          
          {view === 'loading' && (
            <div className="flex flex-col items-center justify-center py-20 text-center relative z-10 my-auto">
              <div className="relative">
                <div className="absolute inset-0 rounded-full blur-xl bg-cyan-500/20 animate-pulse"></div>
                <Loader2 className="w-16 h-16 text-cyan-400 animate-spin mb-8 relative z-10" />
              </div>
              <h2 className="text-3xl font-semibold text-white mb-4">{loadingText}</h2>
              <p className="text-cyan-400 text-lg flex items-center gap-2 animate-pulse">
                <Activity size={20} /> Processing Data...
              </p>
            </div>
          )}

          {view === 'upload' && (
            <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto w-full relative z-10 my-auto">
              {/* Image Upload */}
              <label className="relative flex flex-col items-center justify-center p-10 md:p-16 bg-slate-800 border border-slate-600 rounded-2xl cursor-pointer hover:border-cyan-400 hover:bg-slate-700 transition-all duration-300 group">
                <div className="bg-slate-900 p-6 rounded-full border border-slate-600 shadow-xl mb-6 group-hover:scale-110 transition-all duration-300">
                  <ImageIcon className="w-12 h-12 text-cyan-400" />
                </div>
                <h3 className="text-3xl font-bold text-white mb-3">Search by Image</h3>
                <p className="text-slate-300 text-center text-lg max-w-[300px]">Upload a photo to find this person in the database.</p>
                <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
              </label>

              {/* Video Upload */}
              <label className="relative flex flex-col items-center justify-center p-10 md:p-16 bg-slate-800 border border-slate-600 rounded-2xl cursor-pointer hover:border-blue-400 hover:bg-slate-700 transition-all duration-300 group">
                <div className="bg-slate-900 p-6 rounded-full border border-slate-600 shadow-xl mb-6 group-hover:scale-110 transition-all duration-300">
                  <Video className="w-12 h-12 text-blue-400" />
                </div>
                <h3 className="text-3xl font-bold text-white mb-3">Search by Video</h3>
                <p className="text-slate-300 text-center text-lg max-w-[300px]">Upload a video clip to track and identify a person.</p>
                <input type="file" accept="video/*" className="hidden" onChange={handleVideoUpload} />
              </label>
            </div>
          )}

          {view === 'lineup' && (
            <div className="w-full relative z-10 flex flex-col h-full">
              <div className="text-center mb-10">
                <h2 className="text-4xl font-bold text-white mb-3">Select a Person</h2>
                <p className="text-slate-300 text-xl">We found <span className="text-cyan-400 font-bold px-1">{thumbnails.length}</span> people in the video. Click on the one you want to search for.</p>
              </div>
              
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-6 w-full pb-8">
                {thumbnails.map(thumb => (
                  <button 
                    key={thumb.track_id}
                    onClick={() => handleTargetSelect(thumb.track_id)}
                    className="flex flex-col items-center bg-slate-800 border border-slate-600 rounded-xl overflow-hidden shadow-lg hover:border-cyan-400 hover:bg-slate-700 transition-all duration-300 group p-3 relative hover:-translate-y-1"
                  >
                    <div className="w-full relative overflow-hidden rounded-lg bg-slate-900">
                      <img 
                        src={`data:image/jpeg;base64,${thumb.thumbnail_base64}`} 
                        alt={`Target ${thumb.track_id}`}
                        className="w-full aspect-square object-cover opacity-90 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500"
                      />
                    </div>
                    <div className="mt-4 mb-2 flex items-center justify-center w-full">
                      <span className="font-semibold text-white text-lg">Person #{thumb.track_id}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {view === 'results' && (
            <div className="w-full relative z-10 flex flex-col h-full">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-600 pb-6 mb-8">
                <div>
                  <h2 className="text-4xl font-bold text-white flex items-center gap-3">
                    <Fingerprint className="text-cyan-400" /> Matches Found
                  </h2>
                  <div className="flex flex-wrap gap-2 mt-4">
                    {extractedFeatures.map(f => (
                      <span key={f} className="px-4 py-1.5 bg-slate-700 border border-cyan-500/50 text-cyan-300 text-sm font-semibold capitalize rounded-md">
                        {f} Extracted
                      </span>
                    ))}
                  </div>
                </div>
                <div className="text-slate-200 text-lg mt-4 md:mt-0 font-semibold bg-slate-700 px-5 py-3 rounded-lg border border-slate-600">
                  {matches.length} matches found
                </div>
              </div>

              {matches.length === 0 ? (
                <div className="text-center py-24 text-slate-400 flex flex-col items-center my-auto">
                  <div className="bg-slate-700 p-6 rounded-full mb-6">
                    <CheckCircle2 size={48} className="text-slate-300" />
                  </div>
                  <h3 className="text-2xl text-white font-semibold mb-3">No Results</h3>
                  <p className="text-lg">We couldn't find a strong match for this person in the database.</p>
                </div>
              ) : (
                <div className="relative space-y-8 py-4 w-full">
                  {matches.map((match, idx) => {
                    const { dateStr, startTimeStr, endTimeStr } = formatDateTime(
                      match.metadata?.timestamp || 0,
                      match.metadata?.end_time || match.metadata?.timestamp || 0
                    );
                    
                    const score = match.score * 100;
                    const confidenceColor = score >= 85 ? 'text-green-400 bg-green-900/30 border-green-500/50' : 
                                          score >= 70 ? 'text-yellow-400 bg-yellow-900/30 border-yellow-500/50' : 
                                          'text-red-400 bg-red-900/30 border-red-500/50';

                    return (
                      <div key={idx} className="bg-slate-800 rounded-xl p-6 md:p-8 border border-slate-600 shadow-lg w-full">
                        
                        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4 border-b border-slate-700 pb-6">
                          <div className="flex flex-col">
                            <span className="text-2xl font-bold text-white flex items-center gap-2">
                              {match.metadata?.location || "Unknown Location"}
                            </span>
                            <span className="text-slate-400 mt-2 text-lg">ID: {match.target_id.split("-")[0]}</span>
                          </div>
                          <div className="flex flex-wrap items-center gap-4">
                            <span className={`px-5 py-2 text-lg font-bold rounded-lg border ${confidenceColor}`}>
                              {(score).toFixed(1)}% Match
                            </span>
                            <button
                              onClick={() => handlePlayEvidence(match.target_id)}
                              className={`px-5 py-2 text-lg font-bold rounded-lg transition-all flex items-center gap-2 border ${
                                playingEvidence === match.target_id 
                                  ? 'bg-slate-700 text-white border-slate-500 hover:bg-slate-600' 
                                  : 'bg-cyan-600 text-white border-cyan-500 hover:bg-cyan-500'
                              }`}
                            >
                              {playingEvidence === match.target_id ? (
                                <>Close Video</>
                              ) : (
                                <><Play size={18} className="fill-current" /> Play Video</>
                              )}
                            </button>
                          </div>
                        </div>

                        {/* Time Details */}
                        <div className="bg-slate-900/50 rounded-lg p-6 border border-slate-700 mb-6">
                          <div className="flex justify-between text-base text-slate-300 mb-4 border-b border-slate-700/80 pb-3 font-semibold">
                            <span className="flex items-center gap-2"><Clock size={18} className="text-cyan-400"/> {dateStr}</span>
                            <span>{match.metadata?.camera_id || "Unknown Camera"}</span>
                          </div>
                          <div className="flex justify-between items-center text-lg font-medium">
                            <div className="flex flex-col w-1/3">
                              <span className="text-sm text-slate-400 mb-1">First Seen</span>
                              <span className="text-white text-xl font-bold">{startTimeStr}</span>
                            </div>
                            <div className="w-1/3 flex justify-center text-slate-500">
                              <ChevronRight size={28} />
                            </div>
                            <div className="flex flex-col items-end w-1/3">
                              <span className="text-sm text-slate-400 mb-1">Last Seen</span>
                              <span className="text-white text-xl font-bold">{endTimeStr}</span>
                            </div>
                          </div>
                        </div>

                        {/* Score Breakdown */}
                        {match.metadata?.breakdown && (
                          <div className="mt-6">
                            <div className="text-sm font-semibold text-slate-400 mb-3">Match Details</div>
                            <div className="flex gap-4">
                              {match.metadata.breakdown.face !== undefined && (
                                <div className="bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-2 flex items-center gap-3">
                                  <span className="text-slate-300">Face:</span>
                                  <span className="text-cyan-300 font-bold text-lg">{(match.metadata.breakdown.face * 100).toFixed(1)}%</span>
                                </div>
                              )}
                              {match.metadata.breakdown.appearance !== undefined && (
                                <div className="bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-2 flex items-center gap-3">
                                  <span className="text-slate-300">Clothing:</span>
                                  <span className="text-cyan-300 font-bold text-lg">{(match.metadata.breakdown.appearance * 100).toFixed(1)}%</span>
                                </div>
                              )}
                              {match.metadata.breakdown.gait !== undefined && (
                                <div className="bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-2 flex items-center gap-3">
                                  <span className="text-slate-300">Gait:</span>
                                  <span className="text-cyan-300 font-bold text-lg">{(match.metadata.breakdown.gait * 100).toFixed(1)}%</span>
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* Evidence Video Player */}
                        {playingEvidence === match.target_id && (
                          <div className="mt-8 rounded-xl overflow-hidden border-2 border-slate-600 bg-black flex flex-col w-full relative shadow-2xl">
                            <div className="absolute top-4 right-4 z-20 flex gap-3">
                              <button 
                                onClick={handleTogglePause}
                                className="bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded-lg text-white transition-colors border border-slate-600 flex items-center gap-2 font-bold shadow-lg"
                              >
                                {isEvidencePaused ? <Play size={18} className="fill-current"/> : <Pause size={18} className="fill-current"/>}
                                {isEvidencePaused ? "Play" : "Pause"}
                              </button>
                              <button 
                                onClick={handleRestartEvidence}
                                className="bg-cyan-700 hover:bg-cyan-600 px-4 py-2 rounded-lg text-white transition-colors border border-cyan-500 flex items-center gap-2 font-bold shadow-lg"
                              >
                                <RotateCcw size={18} /> Restart
                              </button>
                            </div>
                            
                            <canvas 
                              id="evidence-canvas" 
                              className={`w-full max-h-[700px] object-contain ${isEvidencePaused ? 'block' : 'hidden'}`} 
                            />
                            <img 
                              id="evidence-img"
                              crossOrigin="anonymous"
                              src={isEvidencePaused ? "" : `${API_BASE}/evidence/${match.target_id}?client_id=${evidenceKey}`} 
                              className={`w-full max-h-[700px] object-contain ${!isEvidencePaused ? 'block' : 'hidden'}`}
                            />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      {/* Global Custom Scrollbar Styling */}
      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(30, 41, 59, 0.5);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(71, 85, 105, 0.8);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(148, 163, 184, 1);
        }
      `}} />
    </div>
  );
}

export default App;
