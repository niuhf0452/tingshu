// Enums mirror `server/app/core/enums.py`. Wire format is the lowercase
// string (`"male"`, `"teen"`, …) so JSON round-trips stay stable.
import Foundation

enum Gender: String, Codable, CaseIterable, Sendable {
    case male
    case female
    case neutral
}

enum Age: String, Codable, CaseIterable, Sendable {
    case child
    case teen
    case youth
    case adult
    case elder
}

enum Personality: String, Codable, CaseIterable, Sendable {
    case calm, gentle, cheerful, serious, cold, fierce
    case determined, timid, playful, mature, naive, wise
    case arrogant, kind, cunning, brave, melancholy, passionate
}

enum Tone: String, Codable, CaseIterable, Sendable {
    case neutral
    case happy
    case sad
    case angry
    case fearful
    case surprised
    case gentle
    case serious
    case playful
    case whisper
}

enum BookStatus: String, Codable, Sendable {
    case uploading
    case processing
    case ready
    case failed
    // Client-only state for books where the server says `ready` but we
    // haven't finished pulling down `meta.json` + chapter texts yet.
    case downloading
}
