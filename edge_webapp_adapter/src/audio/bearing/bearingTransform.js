// src/audio/bearing/bearingTransform.js

/**
 * Convert local bearing (node frame) to global bearing.
 *
 * @param {number} thetaLocalDeg - local bearing [deg]
 * @param {number} headingDeg    - node heading in global frame [deg]
 * @returns {number} global bearing [deg] in [0, 360)
 */
function localToGlobalBearing(thetaLocalDeg, headingDeg) {
    let bearing = thetaLocalDeg + headingDeg;
    bearing = bearing % 360;
    if (bearing < 0) bearing += 360;
    return bearing;
}
